# SPDX-License-Identifier: Apache-2.0
"""Drop-in attention that simulates the served NVFP4-KV + FP8-Q path.

Wrap your model's attention score computation with :class:`NVFP4FakeQuantScores`
(or call :func:`nvfp4_fake_quant_attention` directly).  The forward fake-quantizes
Q -> fp8, K,V -> NVFP4 (matching vLLM exactly), then runs an ordinary BF16 scaled
dot-product attention on the *dequantized* tensors so the softmax / logit error is
identical to what the trtllm-gen kernel produces at serve time.

Per-layer ``k_scale``, ``v_scale``, ``q_scale`` are ``nn.Parameter``s.  Choose
which stage trains them via :meth:`set_trainable` (Stage-1 trains scales only;
Stage-2 also unfreezes projection weights elsewhere).

IMPORTANT wiring rules (else train/inference skew):
  * Apply RoPE to Q and K **before** calling this (K is cached post-RoPE).
  * Do NOT separately multiply by the served ``bmm1_scale`` — quant->dequant
    already reproduces the scaled values; folding the scale twice is wrong.
  * Pass the same ``softmax_scale`` (1/sqrt(head_dim)), causal mask, and any
    logit soft-cap that the served model uses.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nvfp4_qad.fake_quant import fake_quant_kv_nvfp4, fake_quant_q_fp8


def _to_log_scale(x: float | torch.Tensor) -> torch.Tensor:
    """Convert a positive scale value to log space for parameterisation."""
    tiny = torch.finfo(torch.float32).tiny
    return torch.as_tensor(x, dtype=torch.float32).clamp_min(tiny).log()


def nvfp4_fake_quant_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    q_scale: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    is_causal: bool = True,
    attn_mask: torch.Tensor | None = None,
    quantize_q: bool = True,
    return_probs: bool = False,
    subsample_q: int | None = None,
):
    """Run attention with NVFP4 K/V + FP8 Q fake-quant.

    Q/K/V are ``[batch, heads, seq, head_dim]`` (post-RoPE for Q/K).  Returns the
    attention output, and optionally the post-softmax probabilities (for the
    attention-map distillation loss).

    Args:
        subsample_q: if given, restrict the attention map computation to this many
            randomly-selected query positions.  Only applies when
            ``return_probs=True``; has no effect otherwise.  Use this to bound
            memory at long context — the QK^T matrix is ``O(subsample_q * seq)``
            instead of ``O(seq^2)``.
    """
    qd = fake_quant_q_fp8(q, q_scale) if quantize_q else q
    kd = fake_quant_kv_nvfp4(k, k_scale)
    vd = fake_quant_kv_nvfp4(v, v_scale)

    if not return_probs:
        out = F.scaled_dot_product_attention(
            qd, kd, vd, attn_mask=attn_mask, is_causal=is_causal and attn_mask is None,
            scale=softmax_scale,
        )
        return out, None

    # Explicit path so we can hand back the softmax map for distillation.
    # Subsample query positions BEFORE computing QK^T to bound memory at long ctx.
    scale = softmax_scale if softmax_scale is not None else qd.shape[-1] ** -0.5
    seq = qd.shape[-2]

    if subsample_q is not None and subsample_q < seq:
        idx = torch.randperm(seq, device=qd.device)[:subsample_q]
        qd_sub = qd[..., idx, :]
        mask_sub = attn_mask[..., idx, :] if attn_mask is not None else None
        causal_mask_sub = None
        if is_causal and mask_sub is None:
            full_pos = idx.unsqueeze(1)
            k_pos = torch.arange(seq, device=qd.device).unsqueeze(0)
            causal_mask_sub = k_pos > full_pos  # True where masked
    else:
        idx = None
        qd_sub = qd
        mask_sub = attn_mask
        causal_mask_sub = None

    logits = torch.matmul(qd_sub, kd.transpose(-1, -2)) * scale
    if mask_sub is not None:
        logits = logits + mask_sub
    elif causal_mask_sub is not None:
        logits = logits.masked_fill(causal_mask_sub.unsqueeze(0).unsqueeze(0), float("-inf"))
    elif is_causal and idx is None:
        s = logits.shape[-1]
        causal = torch.triu(
            torch.ones(s, s, dtype=torch.bool, device=logits.device), diagonal=1
        )
        logits = logits.masked_fill(causal, float("-inf"))
    probs = logits.softmax(dim=-1)
    out_sub = torch.matmul(probs, vd)

    if idx is not None:
        out = torch.zeros_like(qd)
        out[..., idx, :] = out_sub
    else:
        out = out_sub
    return out, probs


class NVFP4FakeQuantScores(nn.Module):
    """Holds the learnable per-layer scales and applies the fake-quant attention.

    Plug this in place of your attention's score computation.  One instance per
    attention layer.

    Scales are stored in **log space** (``k_scale = exp(log_k_scale)``).  This is
    the standard parametrization for quant scales: scales are tiny (~amax/448, so
    ~1e-2..1e-3) and ``global_scale = 1/k_scale`` amplifies a linear-space
    gradient by ``1/k_scale**2`` -- optimizing the raw scalar explodes.  Log space
    keeps scales strictly positive and the gradient well-conditioned, so a single
    larger LR works across layers.
    """

    def __init__(
        self,
        *,
        k_scale_init: torch.Tensor | float = 1.0,
        v_scale_init: torch.Tensor | float = 1.0,
        q_scale_init: torch.Tensor | float = 1.0,
        softmax_scale: float | None = None,
        quantize_q: bool = True,
    ):
        super().__init__()
        self.log_k_scale = nn.Parameter(_to_log_scale(k_scale_init))
        self.log_v_scale = nn.Parameter(_to_log_scale(v_scale_init))
        self.log_q_scale = nn.Parameter(_to_log_scale(q_scale_init))
        self.softmax_scale = softmax_scale
        self.quantize_q = quantize_q

    # Positive scales materialized from the log-space parameters (keep grad).
    @property
    def k_scale(self) -> torch.Tensor:
        return self.log_k_scale.exp()

    @property
    def v_scale(self) -> torch.Tensor:
        return self.log_v_scale.exp()

    @property
    def q_scale(self) -> torch.Tensor:
        return self.log_q_scale.exp()

    def scale_parameters(self) -> list[nn.Parameter]:
        return [self.log_k_scale, self.log_v_scale, self.log_q_scale]

    def set_trainable(self, scales: bool) -> None:
        """Stage-1 QAD: enable grad on the (log) scales only."""
        for p in self.scale_parameters():
            p.requires_grad_(scales)

    @torch.no_grad()
    def clamp_scales(self) -> None:
        """Keep log-scales in a sane finite range after each optimizer step."""
        for p in self.scale_parameters():
            p.clamp_(min=-30.0, max=30.0)

    def forward(self, q, k, v, *, is_causal=True, attn_mask=None, return_probs=False,
                subsample_q=None):
        return nvfp4_fake_quant_attention(
            q, k, v,
            self.k_scale, self.v_scale, self.q_scale,
            softmax_scale=self.softmax_scale,
            is_causal=is_causal,
            attn_mask=attn_mask,
            quantize_q=self.quantize_q,
            return_probs=return_probs,
            subsample_q=subsample_q,
        )

    def export_scales(self) -> dict[str, float]:
        """Per-tensor positive scalars for the vLLM checkpoint (k/v/q_scale)."""
        result = {}
        for name, scale in (("k_scale", self.k_scale), ("v_scale", self.v_scale),
                             ("q_scale", self.q_scale)):
            val = float(scale.detach().item())
            assert val > 0.0, (
                f"{name}={val} must be > 0; vLLM silently defaults to 1.0 for "
                "missing/non-positive scales"
            )
            result[name] = val
        return result
