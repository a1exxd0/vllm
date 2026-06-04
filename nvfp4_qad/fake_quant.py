# SPDX-License-Identifier: Apache-2.0
"""Differentiable NVFP4 / FP8 fake-quant that bit-matches the vLLM kernel.

Source of truth (this fork):
  * ``csrc/libtorch_stable/quantization/fp4/nvfp4_utils.cuh`` —
    ``cvt_warp_fp16_to_fp4`` (the per-16-element store recipe).
  * ``vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py`` —
    ``ref_nvfp4_quant`` / ``ref_nvfp4_quant_dequant`` (PyTorch reference whose
    forward we reproduce exactly; verified equal to the CUDA kernel by
    :mod:`nvfp4_qad.parity`).
  * ``vllm/v1/attention/backends/flashinfer.py`` — Q is quantized to
    ``fp8_e4m3`` per-tensor for the NVFP4 KV trtllm-gen path; the per-layer
    ``_q_scale`` / ``_k_scale`` / ``_v_scale`` are folded into the attention
    matmul scales.

Key fact: with ``global_scale = 1 / k_scale`` the dequantized value reproduces
the original tensor *nominally* (the ``k_scale`` cancels).  ``k_scale`` only
controls how the per-16 fp8-e4m3 *block* scale lands in range: a bad scale makes
block scales saturate at 448 or underflow to 0, corrupting K/V.  That is exactly
what QAD calibrates / learns, and the fake-quant models it because the forward
clamps the block scale to ``[-448, 448]`` and rounds it through fp8.

The forward here is value-identical to ``ref_nvfp4_quant`` (same E2M1 grid, same
fp8 block-scale round-trip, same clamps).  We only add straight-through
estimators so gradients flow to ``x`` and to the (learnable) scale.
"""

from __future__ import annotations

import torch

# E2M1 (NVFP4) constants — must match scalar_types.float4_e2m1f and the kernel.
NVFP4_BLOCK_SIZE = 16
FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0

# Large sentinel used by _safe_reciprocal: 1/(0 + _SAFE_RECIP_EPS) ≈ 0,
# matching the behaviour of get_reciprocal in nvfp4_emulation_utils.
_SAFE_RECIP_EPS = 1e8


# --------------------------------------------------------------------------- #
# Scale initialization helpers (see nvfp4_qad.calibration for amax collection).
#
# Convention: the checkpoint stores positive per-tensor scalars k/v/q_scale.
# The kernel uses global_scale = 1 / k_scale.  This matches the fork's own
# kernel test (tests/kernels/attention/test_cache.py), which sets
#     k_scale = v_scale = amax / 448   ("Global scale = amax / 448 (per-tensor)")
# With global_scale = 448 / amax, the largest per-16 fp8 block scale lands at
# ~448/6 ~= 74.7 -- comfortably inside the fp8-e4m3 range, leaving headroom and
# avoiding saturation while keeping small blocks above underflow.  Q is plain
# fp8, so q_scale = amax / 448 too.  Re-confirm with nvfp4_qad.parity if a model
# has unusually heavy KV outliers (then prefer a high-quantile amax).
# --------------------------------------------------------------------------- #
def _scale_from_amax(amax: torch.Tensor | float) -> torch.Tensor:
    amax = torch.as_tensor(amax, dtype=torch.float32)
    return (amax / FP8_E4M3_MAX).clamp_min(torch.finfo(torch.float32).tiny)


def init_kv_scale_from_amax(amax: torch.Tensor | float) -> torch.Tensor:
    """Initial per-tensor k_scale / v_scale from an absolute-max estimate."""
    return _scale_from_amax(amax)


def init_q_scale_from_amax(amax: torch.Tensor | float) -> torch.Tensor:
    """Initial per-tensor q_scale from an absolute-max estimate."""
    return _scale_from_amax(amax)


def kv_global_scale(k_scale: torch.Tensor) -> torch.Tensor:
    """Kernel-side ``global_scale = 1 / k_scale`` (the value passed to quant)."""
    return 1.0 / k_scale


# --------------------------------------------------------------------------- #
# Straight-through helpers.
# --------------------------------------------------------------------------- #
def _ste_round(hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
    """Return ``hard`` in the forward pass, ``soft``'s gradient in backward.

    ``hard + (soft - soft.detach())`` ==> value = hard, d/dsoft = 1.
    """
    return hard + (soft - soft.detach())


def _to_fp8_e4m3_ste(x: torch.Tensor) -> torch.Tensor:
    """fp8-e4m3 round-trip with a straight-through gradient (clamped to range)."""
    hard = x.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn).to(torch.float32)
    return _ste_round(hard, x)


def _cast_to_fp4_ste(x: torch.Tensor) -> torch.Tensor:
    """Round to the nearest signed E2M1 value, with a straight-through gradient.

    Forward thresholds match ``cast_to_fp4`` in nvfp4_emulation_utils.py exactly
    (non-mutating, autograd-safe rewrite of the in-place reference).
    """
    sign = torch.sign(x)
    a = x.abs()
    # Nested torch.where reproducing cast_to_fp4's threshold ladder.
    hard = torch.where(
        a > 5.0,
        torch.full_like(a, 6.0),
        torch.where(
            a >= 3.5,
            torch.full_like(a, 4.0),
            torch.where(
                a > 2.5,
                torch.full_like(a, 3.0),
                torch.where(
                    a >= 1.75,
                    torch.full_like(a, 2.0),
                    torch.where(
                        a > 1.25,
                        torch.full_like(a, 1.5),
                        torch.where(
                            a >= 0.75,
                            torch.full_like(a, 1.0),
                            torch.where(
                                a > 0.25,
                                torch.full_like(a, 0.5),
                                torch.zeros_like(a),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    hard = hard * sign
    return _ste_round(hard, x)


def _safe_reciprocal(x: torch.Tensor) -> torch.Tensor:
    """``get_reciprocal`` from nvfp4_emulation_utils (0 -> 0, autograd-safe)."""
    return 1.0 / (x + (x == 0) * _SAFE_RECIP_EPS)


# --------------------------------------------------------------------------- #
# NVFP4 K/V fake-quant.
# --------------------------------------------------------------------------- #
def _nvfp4_quant_dequant_2d(x2d: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
    """Differentiable NVFP4 quant->dequant on a 2D ``[rows, k]`` tensor.

    Forward is value-identical to ``ref_nvfp4_quant`` + its dequant; gradients
    flow to both ``x2d`` and ``global_scale`` (STE through the two roundings).
    ``global_scale`` is a scalar tensor (== ``1 / k_scale``).
    """
    rows, k = x2d.shape
    assert k % NVFP4_BLOCK_SIZE == 0, (
        f"head_size={k} must be divisible by {NVFP4_BLOCK_SIZE}"
    )
    gs = global_scale.to(torch.float32).reshape(())
    xb = x2d.to(torch.float32).reshape(rows, k // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE)

    vec_max = xb.abs().amax(dim=-1, keepdim=True)
    # Block scale: gs * vec_max / 6, clamp to fp8 range, round through fp8-e4m3.
    scale = gs * (vec_max / FP4_E2M1_MAX)
    scale = scale.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    scale = _to_fp8_e4m3_ste(scale)

    output_scale = _safe_reciprocal(scale * _safe_reciprocal(gs))
    scaled = (xb * output_scale).clamp(-FP4_E2M1_MAX, FP4_E2M1_MAX)
    fp4 = _cast_to_fp4_ste(scaled)

    dequant = fp4 * (scale * _safe_reciprocal(gs))
    return dequant.reshape(rows, k).to(x2d.dtype)


def fake_quant_kv_nvfp4(x: torch.Tensor, k_scale: torch.Tensor) -> torch.Tensor:
    """Fake-quantize K or V to NVFP4 along the last (head_size) dimension.

    Args:
        x: ``[..., head_size]`` (e.g. ``[batch, heads, seq, head_size]``).
           For K, pass the tensor **after** RoPE (vLLM caches post-RoPE K).
        k_scale: positive per-tensor scalar (the checkpoint's k_scale / v_scale);
                 may be a learnable ``nn.Parameter`` for Stage-1 QAD.

    Returns the dequantized tensor (same shape/dtype as ``x``), differentiable
    w.r.t. both ``x`` and ``k_scale``.
    """
    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1])
    gs = kv_global_scale(k_scale.to(x.device, torch.float32))
    out = _nvfp4_quant_dequant_2d(x2d, gs)
    return out.reshape(orig_shape)


# --------------------------------------------------------------------------- #
# FP8-e4m3 query fake-quant (the served NVFP4-KV path quantizes Q to fp8).
# --------------------------------------------------------------------------- #
def fake_quant_q_fp8(q: torch.Tensor, q_scale: torch.Tensor) -> torch.Tensor:
    """Per-tensor fp8-e4m3 fake-quant of the query.

    Mirrors flashinfer.py: Q is divided by ``q_scale``, cast to fp8-e4m3, and the
    scale is folded back into ``bmm1_scale`` at serve time (so dequant multiplies
    by ``q_scale``).  ``q_scale`` cancels nominally but governs fp8 saturation.
    Differentiable w.r.t. ``q`` and ``q_scale``.
    """
    qs = q_scale.to(q.device, torch.float32).reshape(())
    scaled = q.to(torch.float32) * _safe_reciprocal(qs)
    deq = _to_fp8_e4m3_ste(scaled) * qs
    return deq.reshape(q.shape).to(q.dtype)
