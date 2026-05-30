# SPDX-License-Identifier: Apache-2.0
"""NVFP4 quantization-aware distillation (QAD) toolkit.

Standalone training-side toolkit (vLLM is inference-only) for fine-tuning a
BF16 model so its attention tolerates the NVFP4 KV cache + FP8 query path that
this vLLM fork serves with ``--kv-cache-dtype nvfp4``.

The fake-quant in :mod:`nvfp4_qad.fake_quant` mirrors vLLM's kernel math
(``cvt_warp_fp16_to_fp4`` / ``ref_nvfp4_quant``) bit-for-bit in the forward
pass, with straight-through estimators on the two hard rounding ops (fp8 block
scale + E2M1 value) so gradients reach both the activations and the learnable
per-layer ``k/v/q_scale``.  :mod:`nvfp4_qad.parity` is the hard gate that this
forward equals the real CUDA kernel on Blackwell (SM100).

See ``README.md`` for the end-to-end recipe.
"""

from nvfp4_qad.fake_quant import (
    NVFP4_BLOCK_SIZE,
    fake_quant_kv_nvfp4,
    fake_quant_q_fp8,
    init_kv_scale_from_amax,
    init_q_scale_from_amax,
    kv_global_scale,
)

__all__ = [
    "NVFP4_BLOCK_SIZE",
    "fake_quant_kv_nvfp4",
    "fake_quant_q_fp8",
    "init_kv_scale_from_amax",
    "init_q_scale_from_amax",
    "kv_global_scale",
]
