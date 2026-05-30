# SPDX-License-Identifier: Apache-2.0
"""Hard parity gate: the fake-quant forward must equal vLLM's quantization.

Two levels (run what your hardware allows):

  1. ``check_against_reference`` (CPU or any CUDA): asserts the differentiable
     :func:`nvfp4_qad.fake_quant.fake_quant_kv_nvfp4` forward equals vLLM's own
     ``ref_nvfp4_quant_dequant`` -- i.e. our STE wrappers don't perturb values.

  2. ``check_against_cuda_kernel`` (Blackwell SM100 only): writes K/V through the
     real ``reshape_and_cache_flash(..., "nvfp4", k_scale, v_scale)`` kernel,
     dequantizes the packed cache with the fork's own test utility, and compares
     to the fake-quant on identical inputs/scales.  This is the gate that proves
     no train/inference skew.

Run:  python -m nvfp4_qad.parity
"""

from __future__ import annotations

import torch

from nvfp4_qad.fake_quant import (
    NVFP4_BLOCK_SIZE,
    fake_quant_kv_nvfp4,
    init_kv_scale_from_amax,
)


def check_against_reference(
    *, rows: int = 512, head_size: int = 128, dtype=torch.bfloat16, seed: int = 0
) -> float:
    """Compare fake-quant vs vLLM ``ref_nvfp4_quant_dequant``. Returns max abs err."""
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
        ref_nvfp4_quant_dequant,
    )

    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, head_size, dtype=dtype, generator=g)
    k_scale = init_kv_scale_from_amax(x.abs().amax())
    global_scale = (1.0 / k_scale).to(torch.float32).reshape(1)

    ref = ref_nvfp4_quant_dequant(x.clone(), global_scale, NVFP4_BLOCK_SIZE)
    mine = fake_quant_kv_nvfp4(x.clone(), k_scale)

    err = (ref.float() - mine.float()).abs().max().item()
    print(f"[ref ] head_size={head_size} dtype={dtype} max|ref-fakequant|={err:.3e}")
    return err


def check_gradients(*, head_size: int = 128) -> None:
    """Sanity-check that gradients reach both the activation and the scale."""
    x = torch.randn(64, head_size, dtype=torch.float32, requires_grad=True)
    k_scale = init_kv_scale_from_amax(x.detach().abs().amax()).clone().requires_grad_(True)
    out = fake_quant_kv_nvfp4(x, k_scale)
    out.pow(2).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all(), "no/NaN grad to x"
    assert k_scale.grad is not None and torch.isfinite(k_scale.grad).all(), (
        "no/NaN grad to k_scale (Stage-1 QAD would not learn)"
    )
    print(f"[grad] grad->x ok (|g|={x.grad.abs().mean():.3e}), "
          f"grad->k_scale ok (g={k_scale.grad.item():.3e})")


def check_against_cuda_kernel(
    *,
    num_blocks: int = 8,
    block_size: int = 16,
    num_heads: int = 8,
    head_size: int = 128,
    num_tokens: int = 64,
    dtype=torch.bfloat16,
    atol: float = 1.5,
    rtol: float = 0.5,
) -> None:
    """SM100-only: fake-quant vs the real CUDA reshape_and_cache_nvfp4 kernel.

    Mirrors tests/kernels/attention/test_cache.py (NHD layout).  Tolerances match
    that test -- NVFP4 is a coarse 4-bit grid, so the gate is that fake-quant and
    kernel agree to the *same* error band against the original tensor, not that
    they are bit-identical (PTX fast-reciprocal vs torch can differ at E2M1
    threshold midpoints).
    """
    import random

    from vllm import _custom_ops as ops
    from vllm.platforms import current_platform
    from vllm.utils.torch_utils import nvfp4_kv_cache_split_views

    if not (current_platform.is_cuda() and current_platform.has_device_capability(100)):
        print("[cuda] SKIP: NVFP4 kernel requires Blackwell (SM100). "
              "Run this on the training/serving GPU.")
        return

    from tests.kernels.attention.test_cache import kv_cache_factory_flashinfer
    from tests.kernels.quantization.nvfp4_utils import dequant_nvfp4_kv_cache

    device = "cuda"
    torch.set_default_device(device)
    num_slots = block_size * num_blocks
    slot_mapping = torch.tensor(
        random.sample(range(num_slots), num_tokens), dtype=torch.long, device=device
    )
    qkv = torch.randn(num_tokens, 3, num_heads, head_size, dtype=dtype, device=device)
    _, key, value = qkv.unbind(dim=1)

    key_caches, value_caches = kv_cache_factory_flashinfer(
        num_blocks, block_size, 1, num_heads, head_size, "nvfp4", dtype,
        device=device, cache_layout="NHD",
    )
    key_cache, value_cache = key_caches[0], value_caches[0]

    k_scale = (key.abs().amax() / 448.0).to(torch.float32)
    v_scale = (value.abs().amax() / 448.0).to(torch.float32)

    ops.reshape_and_cache_flash(
        key, value, key_cache, value_cache, slot_mapping, "nvfp4", k_scale, v_scale
    )

    (k_data,), (k_sf,) = nvfp4_kv_cache_split_views(key_cache)
    (v_data,), (v_sf,) = nvfp4_kv_cache_split_views(value_cache)

    def dequant_nhd(data, sf, gscale):
        res = dequant_nvfp4_kv_cache(
            data.permute(0, 2, 1, 3), sf.permute(0, 2, 1, 3), gscale, head_size, block_size
        )
        return res.permute(0, 2, 1, 3)

    kern_k = dequant_nhd(k_data, k_sf, k_scale.item()).reshape(num_slots, num_heads, head_size)
    kern_v = dequant_nhd(v_data, v_sf, v_scale.item()).reshape(num_slots, num_heads, head_size)
    kern_k = kern_k[slot_mapping].float()
    kern_v = kern_v[slot_mapping].float()

    fq_k = fake_quant_kv_nvfp4(key, k_scale).float()
    fq_v = fake_quant_kv_nvfp4(value, v_scale).float()

    dk = (kern_k - fq_k).abs().max().item()
    dv = (kern_v - fq_v).abs().max().item()
    print(f"[cuda] max|kernel-fakequant|  K={dk:.3e}  V={dv:.3e}")
    # Both must track the original within the same band the kernel test uses.
    torch.testing.assert_close(fq_k, key.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(kern_k, fq_k, atol=atol, rtol=rtol)
    torch.testing.assert_close(kern_v, fq_v, atol=atol, rtol=rtol)
    print("[cuda] PASS: fake-quant matches the CUDA NVFP4 kernel within tolerance.")


if __name__ == "__main__":
    for hs in (16, 64, 128, 256):
        check_against_reference(head_size=hs)
    check_gradients()
    check_against_cuda_kernel()
