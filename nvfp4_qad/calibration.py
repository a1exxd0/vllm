# SPDX-License-Identifier: Apache-2.0
"""Stage-0 calibration: collect per-layer amax(Q/K/V) and init k/v/q_scale.

A running, percentile-or-max absolute-maximum is gathered per attention layer by
feeding a small calibration set through the model with the fake-quant *disabled*
(so we observe the true BF16 activation ranges).  The resulting scales seed the
QAD optimizer; on many models Stage-0 alone recovers most of the accuracy lost
to a naive (scale=1.0) NVFP4 KV cache.

Important: collect K **after** RoPE — that is what vLLM caches and quantizes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from nvfp4_qad.fake_quant import init_kv_scale_from_amax, init_q_scale_from_amax


@dataclass
class AmaxAccumulator:
    """Running absolute-max with an optional high quantile for robustness.

    ``quantile=None`` tracks the true max.  A high quantile (e.g. 0.9999) is more
    robust to rare outliers that would otherwise inflate the scale and push the
    bulk of block scales toward fp8 underflow.
    """

    quantile: float | None = None
    _max: torch.Tensor | None = field(default=None, repr=False)
    _samples: list[torch.Tensor] = field(default_factory=list, repr=False)

    def update(self, x: torch.Tensor) -> None:
        x = x.detach().abs().float()
        if self.quantile is None:
            m = x.amax()
            self._max = m if self._max is None else torch.maximum(self._max, m)
        else:
            # Subsample to bound memory; aggregate quantile at the end.
            flat = x.flatten()
            if flat.numel() > 1_000_000:
                idx = torch.randint(0, flat.numel(), (1_000_000,), device=flat.device)
                flat = flat[idx]
            self._samples.append(flat.cpu())

    def compute(self) -> torch.Tensor:
        if self.quantile is None:
            assert self._max is not None, "AmaxAccumulator received no data"
            return self._max.cpu()
        allv = torch.cat(self._samples)
        return torch.quantile(allv, self.quantile)


@dataclass
class LayerScales:
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    q_scale: torch.Tensor


def attach_calibration_hooks(
    named_attn_modules: dict[str, torch.nn.Module],
    *,
    get_qkv,
    quantile: float | None = None,
) -> tuple[dict, list]:
    """Register forward hooks that accumulate amax(Q/K/V) per attention layer.

    Args:
        named_attn_modules: ``{name: module}`` for every attention block.
        get_qkv: callable ``(module, inputs, output) -> (q, k, v)`` returning the
            **post-RoPE** Q/K and V tensors for that module.  This is
            model-specific; see README for wiring it to your architecture.
        quantile: see :class:`AmaxAccumulator`.

    Returns ``(accumulators, handles)``.  Call :func:`finalize_scales` with the
    accumulators after running calibration data, then ``h.remove()`` each handle.
    """
    accs: dict[str, dict[str, AmaxAccumulator]] = {}
    handles = []
    for name, mod in named_attn_modules.items():
        accs[name] = {
            "q": AmaxAccumulator(quantile),
            "k": AmaxAccumulator(quantile),
            "v": AmaxAccumulator(quantile),
        }

        def _hook(module, inputs, output, _name=name):
            q, k, v = get_qkv(module, inputs, output)
            accs[_name]["q"].update(q)
            accs[_name]["k"].update(k)
            accs[_name]["v"].update(v)

        handles.append(mod.register_forward_hook(_hook))
    return accs, handles


def finalize_scales(accs: dict[str, dict[str, AmaxAccumulator]]) -> dict[str, LayerScales]:
    """Turn accumulated amax into initial per-layer k/v/q scales."""
    out: dict[str, LayerScales] = {}
    for name, a in accs.items():
        out[name] = LayerScales(
            k_scale=init_kv_scale_from_amax(a["k"].compute()),
            v_scale=init_kv_scale_from_amax(a["v"].compute()),
            q_scale=init_q_scale_from_amax(a["q"].compute()),
        )
    return out


@torch.no_grad()
def run_calibration(model, dataloader, accs_handles_setup, *, max_batches: int = 64):
    """Convenience loop: feed ``max_batches`` through the model to fill amax.

    ``accs_handles_setup`` is the ``(accs, handles)`` from
    :func:`attach_calibration_hooks`.  Fake-quant must be *off* during this pass.
    """
    accs, _ = accs_handles_setup
    model.eval()
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        model(**batch)
    return finalize_scales(accs)
