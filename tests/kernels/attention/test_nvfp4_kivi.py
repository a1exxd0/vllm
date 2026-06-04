# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the NVFP4-KIVI attention backend kernel trio.

Tests are gated on CUDA availability (for Triton kernels) and in some cases on
Blackwell (SM100) capability.  On CPU or Ampere hardware the Triton kernel tests
are skipped; ``test_full_block_masks`` runs on any platform.

To run (requires CUDA):
    .venv/bin/python -m pytest tests/kernels/attention/test_nvfp4_kivi.py -v
"""

from __future__ import annotations

import random

import pytest
import torch

from vllm.platforms import current_platform

# The NVFP4-KIVI ops are only available when the KIVI branch is merged.
# Import lazily so the test file itself can always be collected.
try:
    from vllm.v1.attention.ops.triton_nvfp4_kivi import (
        _full_block_masks,
        nvfp4_kivi_gather_dequant,
        nvfp4_kivi_paged_decode,
        nvfp4_kivi_store,
    )
    _KIVI_OPS_AVAILABLE = True
except ImportError:
    _KIVI_OPS_AVAILABLE = False

requires_kivi_ops = pytest.mark.skipif(
    not _KIVI_OPS_AVAILABLE,
    reason="nvfp4_kivi Triton ops not yet merged into this branch",
)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


# --------------------------------------------------------------------------- #
# _full_block_masks (CPU-runnable, no Triton dependency)
# --------------------------------------------------------------------------- #

@requires_kivi_ops
class TestFullBlockMasks:

    def _run(self, slots, block_size):
        sm = torch.tensor(slots, dtype=torch.long)
        starts, full = _full_block_masks(sm, block_size)
        return starts.tolist(), full.tolist()

    def test_empty(self):
        starts, full = self._run([], 16)
        assert starts == []
        assert full == []

    def test_single_token_not_full(self):
        starts, full = self._run([0], 16)
        assert starts == [False]
        assert full == [False]

    def test_exact_one_block(self):
        slots = list(range(16))  # aligned block [0..15]
        starts, full = self._run(slots, 16)
        assert starts[0] is True
        assert all(full)

    def test_partial_block(self):
        slots = list(range(8))  # only half a block
        starts, full = self._run(slots, 16)
        assert not any(full)

    def test_two_aligned_blocks(self):
        slots = list(range(32))
        starts, full = self._run(slots, 16)
        assert starts[0] is True
        assert starts[16] is True
        assert all(full)

    def test_non_aligned_slot(self):
        # Block starting at slot 3 is not page-aligned (3 % 16 != 0).
        slots = list(range(3, 3 + 16))
        starts, full = self._run(slots, 16)
        assert not any(starts)
        assert not any(full)

    def test_gap_in_sequence(self):
        # Consecutive slots except one gap — should NOT be detected as full block.
        slots = list(range(15)) + [16]  # gap at slot 15
        starts, full = self._run(slots, 16)
        assert not any(full)

    @pytest.mark.parametrize("block_size", [16, 32, 64])
    def test_block_size_variants(self, block_size):
        slots = list(range(block_size))
        starts, full = self._run(slots, block_size)
        assert starts[0] is True
        assert all(full)


# --------------------------------------------------------------------------- #
# Round-trip: store -> gather_dequant
# --------------------------------------------------------------------------- #

@requires_kivi_ops
@requires_cuda
@pytest.mark.parametrize("num_heads", [4, 8])
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_store_gather_roundtrip(num_heads, head_size, block_size, dtype):
    """Store K/V to NVFP4 cache, dequant back, check error within quantization band."""
    device = "cuda"
    num_blocks = 4
    num_tokens = block_size  # exactly one full block per head

    torch.manual_seed(0)
    key = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device=device)
    value = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device=device)

    full_dim = head_size // 2 + head_size // 16
    kv_cache = torch.zeros(
        num_blocks, 2, block_size, num_heads, full_dim, dtype=torch.uint8, device=device
    )
    slot_mapping = torch.arange(num_tokens, dtype=torch.long, device=device)

    nvfp4_kivi_store(key, value, kv_cache, slot_mapping, head_size)

    seq_lens = torch.tensor([num_tokens], dtype=torch.int32, device=device)
    block_table = torch.zeros(1, num_blocks, dtype=torch.int32, device=device)
    k_out, v_out = nvfp4_kivi_gather_dequant(
        kv_cache, block_table, seq_lens, head_size, num_heads, num_tokens
    )

    # k_out: [1, H, num_tokens, D] -> [num_tokens, H, D]
    k_rec = k_out[0].permute(1, 0, 2)
    v_rec = v_out[0].permute(1, 0, 2)

    # E2M1 is 4-bit; relative error bound ~50% is generous but realistic.
    k_rel = (k_rec.float() - key.float()).abs() / (key.float().abs() + 1e-6)
    v_rel = (v_rec.float() - value.float()).abs() / (value.float().abs() + 1e-6)
    assert k_rel.median().item() < 0.5, f"K median rel error too large: {k_rel.median():.3f}"
    assert v_rel.median().item() < 0.5, f"V median rel error too large: {v_rel.median():.3f}"


# --------------------------------------------------------------------------- #
# Fused decode vs dense reference
# --------------------------------------------------------------------------- #

@requires_kivi_ops
@requires_cuda
@pytest.mark.parametrize("num_kv_heads,num_q_heads", [(4, 4), (4, 8)])
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("seq_len", [16, 64])
def test_fused_decode_vs_reference(num_kv_heads, num_q_heads, head_size, seq_len):
    """Fused NVFP4-KIVI decode must match the gather+sdpa reference."""
    if not current_platform.is_cuda():
        pytest.skip("CUDA required")

    device = "cuda"
    block_size = 16
    num_blocks = (seq_len + block_size - 1) // block_size
    dtype = torch.bfloat16

    torch.manual_seed(42)
    key = torch.randn(seq_len, num_kv_heads, head_size, dtype=dtype, device=device)
    value = torch.randn(seq_len, num_kv_heads, head_size, dtype=dtype, device=device)
    query = torch.randn(1, num_q_heads, head_size, dtype=dtype, device=device)

    full_dim = head_size // 2 + head_size // 16
    kv_cache = torch.zeros(
        num_blocks, 2, block_size, num_kv_heads, full_dim, dtype=torch.uint8, device=device
    )
    slot_mapping = torch.arange(seq_len, dtype=torch.long, device=device)
    nvfp4_kivi_store(key, value, kv_cache, slot_mapping, head_size)

    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)
    sm_scale = head_size ** -0.5

    # Reference: dequant + SDPA
    k_dense, v_dense = nvfp4_kivi_gather_dequant(
        kv_cache, block_table, seq_lens, head_size, num_kv_heads, seq_len
    )
    k_flat = k_dense[0].permute(1, 0, 2)  # [seq_len, H_k, D]
    v_flat = v_dense[0].permute(1, 0, 2)  # [seq_len, H_k, D]
    q_t = query.permute(1, 0, 2).unsqueeze(0)   # [1, H_q, 1, D]
    k_t = k_flat.permute(1, 0, 2).unsqueeze(0)  # [1, H_k, seq_len, D]
    v_t = v_flat.permute(1, 0, 2).unsqueeze(0)  # [1, H_k, seq_len, D]
    ref_out = torch.nn.functional.scaled_dot_product_attention(
        q_t, k_t, v_t, scale=sm_scale, enable_gqa=(num_kv_heads < num_q_heads)
    ).squeeze(0).permute(1, 0, 2)  # [1, H_q, D]

    # Fused decode
    fused_out = nvfp4_kivi_paged_decode(query, kv_cache, block_table, seq_lens, sm_scale)

    # Allow ~1e-2 relative tolerance for the combined quant + bf16 computation.
    torch.testing.assert_close(
        fused_out.float(), ref_out.float(), atol=2e-2, rtol=2e-2,
        msg="fused decode diverged from gather+SDPA reference",
    )
