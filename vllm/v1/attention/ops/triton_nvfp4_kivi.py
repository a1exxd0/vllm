# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged NVFP4-KIVI store / dequant Triton kernels for the vLLM backend.

This is the **NVFP4 sibling** of ``triton_int4_kivi.py``.  It reuses that file's
paged layout, KIVI per-channel-K / per-token-V geometry, sync-free store and
fused split-K flash-decode *verbatim*; the ONLY thing that changes is the 4-bit
number format used for each element:

  * INT4-KIVI : symmetric integer codes in ``[-7, 7]``; deq = ``code * scale``
                (uniform grid).
  * NVFP4-KIVI: 4-bit float ``E2M1`` (1 sign, 2 exp, 1 mantissa) whose magnitudes
                are ``{0, .5, 1, 1.5, 2, 3, 4, 6}`` (non-uniform grid, denser near
                zero); deq = ``e2m1_decode(code) * scale``.

Because both formats are 4 bits and both use a per-16-element ``fp8_e4m3`` block
scale, the packed cache tensor is **bit-budget identical** to int4 (so the same
~3.2x memory win holds) and the same paged kernels apply with only the
encode/decode swapped.  We deliberately keep the SAME MSE α-clip calibration as
int4 (grid-search α in [0.5,1.0]); the sole independent variable vs INT4-KIVI is
the grid, so an A/B isolates "E2M1 vs uniform-int4 at 4 bits" for KV cache.

NVFP4 normally carries a per-tensor fp32 *global* scale on top of the fp8 block
scale; for a KV cache the block scale alone (amax/6 rounded to fp8_e4m3, whose
range 2^-9..448 comfortably covers K/V block amaxes) already spans the dynamic
range, so we fold it away and store exactly one fp8 byte per 16-block — keeping
the cache identical to int4.  ``code = sign<<3 | e2m1_magnitude_index``; magnitude
index 0..7 maps to the 8 E2M1 magnitudes above.

PAGED CACHE LAYOUT (uint8), one tensor per layer (identical to INT4-KIVI):
    kv_cache[num_blocks, 2, block_size, num_kv_heads, full_dim]
      dim 1: 0 = K side, 1 = V side
      full_dim = head_size//2 (nibble-packed E2M1 data) + head_size//16 (scales)
    Within a token row: [ data_bytes | scale_bytes ].
    Scales are fp8_e4m3 (1 byte / 16-elem block), stored as uint8, reinterpreted
    as fp8 by the kernels.

K is stored per-channel for every FULL 16-token block (one scale per channel over
the block's 16 tokens) and per-token for partial trailing blocks; V is always
per-token.  The full/partial decision is purely geometric (block ``b`` is full iff
``(b+1)*16 <= L``) — see ``triton_int4_kivi`` for the full rationale.
"""

from __future__ import annotations

import functools
import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

BLOCK = 16  # elements per quant block (along head_dim)
PACK = BLOCK // 2  # 8 packed bytes per 16-elem block
N_MSE = 16  # MSE clip-search grid points (alpha in [0.5, 1.0])
FP4_MAX = 6.0  # max representable E2M1 magnitude

_BLOCK = tl.constexpr(BLOCK)
_PACK = tl.constexpr(PACK)
_N_MSE = tl.constexpr(N_MSE)
_FP4_MAX = tl.constexpr(FP4_MAX)

# ---- Fused-decode tuning knobs (env-overridable for sweeps).  Mirror the int4
# knobs but under their own VLLM_NVFP4_DECODE_* names so the two backends can be
# tuned independently.  Defaults copy the int4 B300 (sm_103) optimum. ----
_DECODE_BLOCK_N = int(os.environ.get("VLLM_NVFP4_DECODE_BLOCK_N", "64"))
_DECODE_NUM_WARPS = int(os.environ.get("VLLM_NVFP4_DECODE_WARPS", "4"))
_DECODE_NUM_STAGES = int(os.environ.get("VLLM_NVFP4_DECODE_STAGES", "3"))
_DECODE_WAVES = float(os.environ.get("VLLM_NVFP4_DECODE_WAVES", "4"))
_DECODE_MAX_SPLIT = int(os.environ.get("VLLM_NVFP4_DECODE_MAX_SPLIT", "64"))


@functools.lru_cache(maxsize=None)
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


# --------------------------------------------------------------------------- #
# E2M1 (NVFP4) encode / decode primitives.
# --------------------------------------------------------------------------- #
@triton.jit
def _e2m1_decode_mag(mag):
    """3-bit magnitude index (0..7) -> E2M1 float magnitude, binary-tree LUT.

    Maps to {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} (mirrors the reference
    ``nvfp4_emulation_utils._e2m1_inline``).
    """
    b2 = (mag >> 2) & 1
    b1 = (mag >> 1) & 1
    b0 = mag & 1
    low = tl.where(b1 == 1, tl.where(b0 == 1, 1.5, 1.0),
                   tl.where(b0 == 1, 0.5, 0.0))
    high = tl.where(b1 == 1, tl.where(b0 == 1, 6.0, 4.0),
                    tl.where(b0 == 1, 3.0, 2.0))
    return tl.where(b2 == 1, high, low)


@triton.jit
def _e2m1_encode(x, scale):
    """Quantize ``x`` against ``scale`` to a 4-bit E2M1 code ``sign<<3 | mag``.

    Round-to-nearest using the midpoints between adjacent E2M1 magnitudes
    (identical thresholds to the reference ``cast_to_fp4``).  Returns int32 codes
    in ``[0, 15]``.
    """
    y = libdevice.div_rn(x, scale)
    a = tl.abs(y)
    idx = tl.where(a < 0.25, 0,
          tl.where(a < 0.75, 1,
          tl.where(a < 1.25, 2,
          tl.where(a < 1.75, 3,
          tl.where(a < 2.5, 4,
          tl.where(a < 3.5, 5,
          tl.where(a < 5.0, 6, 7))))))).to(tl.int32)
    sign = tl.where(y < 0.0, 8, 0).to(tl.int32)
    return sign | idx


@triton.jit
def _recon_e2m1(x, scale):
    """Quantize-then-dequantize ``x`` through E2M1 at ``scale`` (for MSE search)."""
    code = _e2m1_encode(x, scale)
    val = _e2m1_decode_mag(code & 7)
    return tl.where(((code >> 3) & 1) == 1, -val, val) * scale


@triton.jit
def _mse_scale(x, amax):
    """MSE-optimal clip scale for a 1-D E2M1 block: grid-search alpha in
    [0.5,1.0] minimising sum((x - deq)^2).  Same calibration shape as int4's
    ``_mse_scale`` but with E2M1 reconstruction (alpha=1 == plain amax/FP4_MAX)."""
    best_err = 1e38
    best_scale = libdevice.div_rn(amax, _FP4_MAX)
    for i in tl.static_range(_N_MSE):
        a = 0.5 + i * (0.5 / (_N_MSE - 1))
        s = libdevice.div_rn(a * amax, _FP4_MAX)
        deq = _recon_e2m1(x, s)
        diff = x - deq
        err = tl.sum(diff * diff)
        take = err < best_err
        best_err = tl.where(take, err, best_err)
        best_scale = tl.where(take, s, best_scale)
    return best_scale


@triton.jit
def _mse_scale_axis0(x, amax):
    """Per-channel E2M1 MSE-optimal clip.  ``x`` is [16 tokens, D], ``amax`` is
    [D] (per-channel absmax over the token axis).  Returns [D]."""
    best_err = tl.full(amax.shape, 1e38, tl.float32)
    best_scale = libdevice.div_rn(amax, _FP4_MAX)
    for i in tl.static_range(_N_MSE):
        a = 0.5 + i * (0.5 / (_N_MSE - 1))
        s = libdevice.div_rn(a * amax, _FP4_MAX)  # [D]
        deq = _recon_e2m1(x, s[None, :])  # [16, D]
        diff = x - deq
        err = tl.sum(diff * diff, axis=0)  # [D]
        take = err < best_err
        best_err = tl.where(take, err, best_err)
        best_scale = tl.where(take, s, best_scale)
    return best_scale


@triton.jit
def _store_token_kernel(
    src_ptr,  # bf16 [N, H, D]
    cache_ptr,  # uint8 [num_blocks, 2, block_size, H, full_dim]
    slot_ptr,  # int64 [N]
    N,
    H: tl.constexpr,
    D: tl.constexpr,
    ND: tl.constexpr,  # D // BLOCK
    DATA_BYTES: tl.constexpr,  # D // 2
    FULL_DIM: tl.constexpr,  # DATA_BYTES + ND
    KV_SIDE: tl.constexpr,  # 0 = K, 1 = V
    BLOCK_SIZE: tl.constexpr,
    s_src_n,
    s_src_h,
    s_cache_blk,
    s_cache_side,
    s_cache_tok,
    s_cache_h,
):
    """One program per (token, head, dblock).  Quantize 16 head_dim elements."""
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_db = tl.program_id(2)

    slot = tl.load(slot_ptr + pid_n)
    if slot < 0:
        return
    blk_idx = slot // BLOCK_SIZE
    tok_in_blk = slot % BLOCK_SIZE

    ch = pid_db * _BLOCK + tl.arange(0, _BLOCK)  # [BLOCK]
    src_off = pid_n * s_src_n + pid_h * s_src_h + ch
    x = tl.load(src_ptr + src_off).to(tl.float32)  # [BLOCK]

    amax = tl.maximum(tl.max(tl.abs(x)), 1e-9)
    scale = _mse_scale(x, amax)
    # Round scale to fp8_e4m3 BEFORE encoding so the codes match what dequant
    # reads back (store/dequant bit-consistent, no double-rounding mismatch).
    scale_byte = scale.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    scale = scale_byte.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    codes = _e2m1_encode(x, scale)  # [BLOCK] in [0, 15]

    # pack: elem 2j -> low nibble of byte j, 2j+1 -> high nibble
    ci = codes & 0xF  # [BLOCK]
    nib2 = tl.reshape(ci, (_PACK, 2))
    lo, hi = tl.split(nib2)  # each [PACK]
    packed = (lo | (hi << 4)).to(tl.uint8)  # [PACK]

    row_base = (
        blk_idx * s_cache_blk
        + KV_SIDE * s_cache_side
        + tok_in_blk * s_cache_tok
        + pid_h * s_cache_h
    )
    data_off = row_base + pid_db * _PACK + tl.arange(0, _PACK)
    tl.store(cache_ptr + data_off, packed)

    s_off = row_base + DATA_BYTES + pid_db
    tl.store(cache_ptr + s_off, scale_byte)


@triton.jit
def _store_k_channel_kernel(
    src_ptr,  # bf16 [N, H, D]
    cache_ptr,  # uint8 [num_blocks, 2, block_size, H, full_dim]
    slot_ptr,  # int64 [N]  slot of each src row
    fbs_ptr,  # uint8 [N]  1 iff src row starts a full, contiguous, aligned block
    H: tl.constexpr,
    D: tl.constexpr,
    ND: tl.constexpr,  # D // BLOCK
    DATA_BYTES: tl.constexpr,  # D // 2
    BLOCK_SIZE: tl.constexpr,
    s_src_n,
    s_src_h,
    s_cache_blk,
    s_cache_side,
    s_cache_tok,
    s_cache_h,
):
    """One program per (src_row, head).  K per-channel E2M1 over 16 tokens.

    Identical control flow to the int4 per-channel store; only the quant math is
    E2M1.  Data nibbles are packed per token (same layout as per-token) and the
    D per-channel fp8 scales are written into the block's K-scale region.
    """
    pid_r = tl.program_id(0)
    pid_h = tl.program_id(1)

    if tl.load(fbs_ptr + pid_r) == 0:
        return  # not a full-block start -> these tokens go via the per-token path

    row0 = pid_r
    phys_blk = tl.load(slot_ptr + pid_r) // BLOCK_SIZE

    tok = tl.arange(0, BLOCK_SIZE)  # [16] token-in-block == row offset
    ch = tl.arange(0, D)  # [D]
    src_off = (row0 + tok)[:, None] * s_src_n + pid_h * s_src_h + ch[None, :]
    x = tl.load(src_ptr + src_off).to(tl.float32)  # [16, D]

    amax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-9)  # [D] over the 16 tokens
    scale = _mse_scale_axis0(x, amax)  # [D]
    scale_byte = scale.to(tl.float8e4nv).to(tl.uint8, bitcast=True)  # [D]
    scale = scale_byte.to(tl.float8e4nv, bitcast=True).to(tl.float32)  # [D]
    codes = _e2m1_encode(x, scale[None, :])  # [16, D] in [0, 15]

    blk_base = (
        phys_blk * s_cache_blk
        + 0 * s_cache_side  # K side
        + pid_h * s_cache_h
    )

    ci = codes & 0xF  # [16, D]
    nib2 = tl.reshape(ci, (BLOCK_SIZE, DATA_BYTES, 2))  # [16, DATA_BYTES, 2]
    lo, hi = tl.split(nib2)  # each [16, DATA_BYTES]
    packed = (lo | (hi << 4)).to(tl.uint8)  # [16, DATA_BYTES]
    pcol = tl.arange(0, DATA_BYTES)
    data_off = blk_base + tok[:, None] * s_cache_tok + pcol[None, :]
    tl.store(cache_ptr + data_off, packed)

    # channel c -> token (c // ND), scale-byte (c % ND) within the scale region.
    s_tok = ch // ND  # [D]
    s_byte = ch % ND  # [D]
    s_off = blk_base + s_tok * s_cache_tok + DATA_BYTES + s_byte
    tl.store(cache_ptr + s_off, scale_byte)


@triton.jit
def _gather_dequant_kernel(
    cache_ptr,  # uint8 [num_blocks, 2, block_size, H, full_dim]
    block_table_ptr,  # int32 [B, max_blocks]
    seq_lens_ptr,  # int32 [B]
    out_ptr,  # bf16 [B, H, max_seq, D]
    B,
    H: tl.constexpr,
    D: tl.constexpr,
    ND: tl.constexpr,
    DATA_BYTES: tl.constexpr,
    FULL_DIM: tl.constexpr,
    KV_SIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_SEQ,
    s_bt,
    s_cache_blk,
    s_cache_side,
    s_cache_tok,
    s_cache_h,
    s_out_b,
    s_out_h,
    s_out_s,
):
    """One program per (req, pos, head).  Dequant full head_dim of one token."""
    pid_pos = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    seq_len = tl.load(seq_lens_ptr + pid_b)
    if pid_pos >= seq_len:
        return

    logical_blk = pid_pos // BLOCK_SIZE
    tok_in_blk = pid_pos % BLOCK_SIZE
    phys_blk = tl.load(block_table_ptr + pid_b * s_bt + logical_blk).to(tl.int64)

    blk_base = (
        phys_blk * s_cache_blk
        + KV_SIDE * s_cache_side
        + pid_h * s_cache_h
    )
    row_base = blk_base + tok_in_blk * s_cache_tok
    out_base = pid_b.to(tl.int64) * s_out_b + pid_h * s_out_h + pid_pos * s_out_s

    k_per_channel = (KV_SIDE == 0) and (
        (logical_blk + 1) * BLOCK_SIZE <= seq_len
    )

    for db in tl.static_range(ND):
        data_off = row_base + db * _PACK + tl.arange(0, _PACK)
        packed = tl.load(cache_ptr + data_off).to(tl.int32)  # [PACK]
        lo = packed & 0xF
        hi = (packed >> 4) & 0xF
        nib = tl.interleave(lo, hi)  # [BLOCK] codes 0..15
        val = _e2m1_decode_mag(nib & 7)
        codes = tl.where(((nib >> 3) & 1) == 1, -val, val)  # [BLOCK] signed E2M1

        ch = db * _BLOCK + tl.arange(0, _BLOCK)  # [BLOCK] channel indices
        if k_per_channel:
            s_addr = blk_base + (ch // ND) * s_cache_tok + DATA_BYTES + (ch % ND)
            sb_pc = tl.load(cache_ptr + s_addr).to(tl.uint8)  # [BLOCK]
            scale = sb_pc.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        else:
            sb_pt = tl.load(cache_ptr + row_base + DATA_BYTES + db).to(tl.uint8)
            scale = tl.broadcast_to(
                sb_pt.to(tl.float8e4nv, bitcast=True).to(tl.float32)[None], [_BLOCK]
            )

        deq = codes * scale  # [BLOCK]
        tl.store(out_ptr + out_base + ch, deq.to(tl.bfloat16))


def _full_block_masks(slot_mapping: torch.Tensor, block_size: int):
    """Sync-free full-block detection (identical to the int4 implementation)."""
    N = slot_mapping.shape[0]
    dev = slot_mapping.device
    if N < block_size:
        z = torch.zeros(N, dtype=torch.bool, device=dev)
        return z, z
    sm = slot_mapping
    bs = block_size
    idx = torch.arange(N, device=dev)
    aligned = (sm % bs == 0) & (sm >= 0) & (idx + bs <= N)

    adj = (sm[1:] == sm[:-1] + 1).to(torch.int32)  # [N-1]
    cadj = torch.zeros(N, dtype=torch.int32, device=dev)
    torch.cumsum(adj, dim=0, out=cadj[1:])
    run_ok = torch.zeros(N, dtype=torch.bool, device=dev)
    hi = cadj[bs - 1 : N]
    lo = cadj[0 : N - bs + 1]
    run_ok[: N - bs + 1] = (hi - lo) == (bs - 1)
    full_block_start = aligned & run_ok

    si = full_block_start.to(torch.int32)
    csi = torch.zeros(N + 1, dtype=torch.int32, device=dev)
    torch.cumsum(si, dim=0, out=csi[1:])
    lo_i = torch.clamp(idx - bs + 1, min=0)
    full_mask = (csi[idx + 1] - csi[lo_i]) > 0
    return full_block_start, full_mask


def nvfp4_kivi_store(
    key: torch.Tensor,  # bf16 [N, H, D]
    value: torch.Tensor,  # bf16 [N, H, D]
    kv_cache: torch.Tensor,  # uint8 [num_blocks, 2, block_size, H, full_dim]
    slot_mapping: torch.Tensor,  # int64 [N]
    head_size: int,
) -> None:
    """Quantize new K/V tokens to NVFP4 (E2M1) and store into the paged cache.

    V per-token; K per-channel for full 16-token blocks, per-token for the rest.
    """
    N, H, D = key.shape
    if N == 0:
        return
    ND = D // BLOCK
    DATA_BYTES = D // 2
    FULL_DIM = DATA_BYTES + ND
    BLOCK_SIZE = kv_cache.shape[2]

    s_cache_blk = kv_cache.stride(0)
    s_cache_side = kv_cache.stride(1)
    s_cache_tok = kv_cache.stride(2)
    s_cache_h = kv_cache.stride(3)

    # --- V: always per-token ---
    v = value.contiguous()
    _store_token_kernel[(N, H, ND)](
        v, kv_cache, slot_mapping, N,
        H=H, D=D, ND=ND, DATA_BYTES=DATA_BYTES, FULL_DIM=FULL_DIM,
        KV_SIDE=1, BLOCK_SIZE=BLOCK_SIZE,
        s_src_n=v.stride(0), s_src_h=v.stride(1),
        s_cache_blk=s_cache_blk, s_cache_side=s_cache_side,
        s_cache_tok=s_cache_tok, s_cache_h=s_cache_h,
    )

    # --- K: per-channel for full blocks, per-token for the rest ---
    k = key.contiguous()
    full_block_start, full_mask = _full_block_masks(slot_mapping, BLOCK_SIZE)

    if os.environ.get("VLLM_NVFP4_DEBUG"):
        nfull = int(full_block_start.sum().item())
        npart = int((~full_mask).sum().item())
        print(
            f"[nvfp4_kivi.store] N={N} full_blocks={nfull} "
            f"per_channel_rows={int(full_mask.sum().item())} "
            f"per_token_rows={npart}",
            flush=True,
        )

    _store_k_channel_kernel[(N, H)](
        k, kv_cache, slot_mapping, full_block_start.to(torch.uint8),
        H=H, D=D, ND=ND, DATA_BYTES=DATA_BYTES, BLOCK_SIZE=BLOCK_SIZE,
        s_src_n=k.stride(0), s_src_h=k.stride(1),
        s_cache_blk=s_cache_blk, s_cache_side=s_cache_side,
        s_cache_tok=s_cache_tok, s_cache_h=s_cache_h,
    )

    k_slots = slot_mapping.masked_fill(full_mask, -1)
    _store_token_kernel[(N, H, ND)](
        k, kv_cache, k_slots, N,
        H=H, D=D, ND=ND, DATA_BYTES=DATA_BYTES, FULL_DIM=FULL_DIM,
        KV_SIDE=0, BLOCK_SIZE=BLOCK_SIZE,
        s_src_n=k.stride(0), s_src_h=k.stride(1),
        s_cache_blk=s_cache_blk, s_cache_side=s_cache_side,
        s_cache_tok=s_cache_tok, s_cache_h=s_cache_h,
    )


# =========================================================================== #
# Fused paged flash-decode over the packed NVFP4-KIVI cache.  Structure is
# identical to the int4 fused decode; only the K/V unpack (E2M1 instead of signed
# int4) differs.  K/V dequant in registers, bf16 tensor-core dots, online softmax,
# split-K with sync-free auto split + WRITE_FINAL no-combine fast path.
# =========================================================================== #
@triton.jit
def _paged_decode_kernel(
    q_ptr,  # bf16 [B, n_qh, D]
    cache_ptr,  # uint8 [num_blocks, 2, block_size, H, full_dim]
    block_table_ptr,  # int32 [B, max_blocks]
    seq_lens_ptr,  # int32 [B]
    pm_ptr,  # fp32 [B, H, GROUP, SPLIT]
    pl_ptr,  # fp32 [B, H, GROUP, SPLIT]
    pacc_ptr,  # fp32 [B, H, GROUP, SPLIT, D]
    out_ptr,  # bf16 [B*n_qh, D]  (written directly when WRITE_FINAL)
    sm_scale,
    n_qh,
    GROUP: tl.constexpr,
    GPAD: tl.constexpr,  # next-pow2(GROUP), >=16 for tl.dot
    D: tl.constexpr,
    H: tl.constexpr,
    ND: tl.constexpr,  # D // BLOCK
    DATA_BYTES: tl.constexpr,  # D // 2
    BLOCK_SIZE: tl.constexpr,  # paged block (page) size in tokens
    SPLIT: tl.constexpr,
    BLOCK_N: tl.constexpr,  # tokens streamed per inner step
    s_cache_blk,
    s_cache_side,
    s_cache_tok,
    s_cache_h,
    s_bt,
    WRITE_FINAL: tl.constexpr,  # SPLIT==1: write final out, skip partials+combine
):
    pid = tl.program_id(0)
    s = pid % SPLIT
    tmp = pid // SPLIT
    kvh = tmp % H
    b = tmp // H
    qh0 = kvh * GROUP

    d = tl.arange(0, D)  # [D] head-dim lanes
    db = d // _BLOCK  # [D] dblock of each lane
    cin = d % _BLOCK  # [D] in-block channel
    dbyte = db * _PACK + cin // 2  # [D] data byte holding lane d
    dhi = (cin % 2) == 1  # [D] lane d in high nibble?

    gr = tl.arange(0, GPAD)  # [GPAD] query heads within the group
    gmask = gr < GROUP
    qoff = (b * n_qh + qh0 + gr[:, None]) * D + d[None, :]
    q = tl.load(q_ptr + qoff, mask=gmask[:, None], other=0.0).to(tl.bfloat16)

    seq_len = tl.load(seq_lens_ptr + b)
    seg = (seq_len + SPLIT - 1) // SPLIT
    seg0 = s * seg
    seg1 = tl.minimum(seg0 + seg, seq_len)

    m_i = tl.full([GPAD], -float("inf"), tl.float32)
    l_i = tl.zeros([GPAD], tl.float32)
    acc = tl.zeros([GPAD, D], tl.float32)

    t = seg0
    while t < seg1:
        tok = t + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        tmask = tok < seg1
        logical_blk = tok // BLOCK_SIZE
        tok_in_blk = tok % BLOCK_SIZE
        phys = tl.load(
            block_table_ptr + b * s_bt + logical_blk, mask=tmask, other=0
        ).to(tl.int64)  # [BLOCK_N]
        is_full = (logical_blk + 1) * BLOCK_SIZE <= seq_len  # [BLOCK_N]

        rowK = phys * s_cache_blk + tok_in_blk.to(tl.int64) * s_cache_tok + kvh * s_cache_h
        rowV = rowK + s_cache_side
        blkK = phys * s_cache_blk + kvh * s_cache_h  # block base (K side, scales)

        # ---- dequant K -> [D, BLOCK_N] (E2M1) ----
        koff = rowK[None, :] + dbyte[:, None]  # [D, BLOCK_N]
        kb = tl.load(cache_ptr + koff, mask=tmask[None, :], other=0).to(tl.int32)
        knib = tl.where(dhi[:, None], (kb >> 4) & 0xF, kb & 0xF)
        kval = _e2m1_decode_mag(knib & 7)
        kcode = tl.where(((knib >> 3) & 1) == 1, -kval, kval)  # [D, BLOCK_N] signed
        # scale: per-channel for full blocks, per-token for the partial tail.
        ks_pc = blkK[None, :] + (d // ND)[:, None] * s_cache_tok + DATA_BYTES + (d % ND)[:, None]
        ks_pt = rowK[None, :] + DATA_BYTES + db[:, None]
        ksaddr = tl.where(is_full[None, :], ks_pc, ks_pt)
        ksb = tl.load(cache_ptr + ksaddr, mask=tmask[None, :], other=0).to(tl.uint8)
        ksc = ksb.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        # Dequant in fp32, then cast to bf16 so QK^T runs on bf16 tensor cores
        # (fp32 accumulate) — the dominant speedup lever, same as int4.
        kdeq = (kcode * ksc).to(tl.bfloat16)  # [D, BLOCK_N]

        qk = tl.dot(q, kdeq, out_dtype=tl.float32) * sm_scale  # [GPAD, BLOCK_N]
        qk = tl.where(tmask[None, :], qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))  # [GPAD]
        alpha = tl.exp(m_i - m_new)
        pblk = tl.exp(qk - m_new[:, None])
        pblk = tl.where(tmask[None, :], pblk, 0.0)  # [GPAD, BLOCK_N]

        # ---- dequant V -> [BLOCK_N, D] (E2M1, always per-token) ----
        voff = rowV[:, None] + dbyte[None, :]  # [BLOCK_N, D]
        vb = tl.load(cache_ptr + voff, mask=tmask[:, None], other=0).to(tl.int32)
        vnib = tl.where(dhi[None, :], (vb >> 4) & 0xF, vb & 0xF)
        vval = _e2m1_decode_mag(vnib & 7)
        vcode = tl.where(((vnib >> 3) & 1) == 1, -vval, vval)  # [BLOCK_N, D]
        vsaddr = rowV[:, None] + DATA_BYTES + db[None, :]  # [BLOCK_N, D]
        vsb = tl.load(cache_ptr + vsaddr, mask=tmask[:, None], other=0).to(tl.uint8)
        vsc = vsb.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        vdeq = (vcode * vsc).to(tl.bfloat16)  # [BLOCK_N, D]

        acc = acc * alpha[:, None] + tl.dot(
            pblk.to(tl.bfloat16), vdeq, out_dtype=tl.float32
        )  # [GPAD, D]
        l_i = l_i * alpha + tl.sum(pblk, axis=1)
        m_i = m_new
        t += BLOCK_N

    if WRITE_FINAL:
        l_safe = tl.where(l_i > 0.0, l_i, 1.0)  # empty-request guard
        o = acc / l_safe[:, None]
        orow = b * n_qh + qh0 + gr  # [GPAD] == output row (b, qh)
        tl.store(out_ptr + orow[:, None] * D + d[None, :], o.to(tl.bfloat16),
                 mask=gmask[:, None])
    else:
        base = ((b * H + kvh) * GROUP + gr) * SPLIT + s  # [GPAD]
        tl.store(pm_ptr + base, m_i, mask=gmask)
        tl.store(pl_ptr + base, l_i, mask=gmask)
        tl.store(pacc_ptr + base[:, None] * D + d[None, :], acc, mask=gmask[:, None])


@triton.jit
def _decode_combine_kernel(
    pm_ptr,
    pl_ptr,
    pacc_ptr,
    out_ptr,  # bf16 [B*n_qh, D]
    D: tl.constexpr,
    SPLIT: tl.constexpr,
):
    pid = tl.program_id(0)  # over B*n_qh
    d = tl.arange(0, D)
    m = -float("inf")
    for sp in range(0, SPLIT):
        m = tl.maximum(m, tl.load(pm_ptr + pid * SPLIT + sp))
    l = 0.0
    acc = tl.zeros([D], dtype=tl.float32)
    for sp in range(0, SPLIT):
        ms = tl.load(pm_ptr + pid * SPLIT + sp)
        ls = tl.load(pl_ptr + pid * SPLIT + sp)
        a = tl.load(pacc_ptr + (pid * SPLIT + sp) * D + d)
        scale = tl.exp(ms - m)
        acc += a * scale
        l += ls * scale
    l = tl.where(l > 0.0, l, 1.0)  # empty request guard (l==0 -> out 0)
    tl.store(out_ptr + pid * D + d, (acc / l).to(tl.bfloat16))


def nvfp4_kivi_paged_decode(
    q: torch.Tensor,  # bf16 [N, Hq, D], N == B (one decode token per request)
    kv_cache: torch.Tensor,  # uint8 [num_blocks, 2, block_size, H, full_dim]
    block_table: torch.Tensor,  # int32 [B, max_blocks]
    seq_lens: torch.Tensor,  # int32 [B]
    sm_scale: float,
    split: int | None = None,
    block_n: int | None = None,
) -> torch.Tensor:
    """Fused NVFP4-KIVI flash-decode: attention straight off the packed cache."""
    N, Hq, D = q.shape
    B = seq_lens.shape[0]
    assert N == B, "paged decode expects exactly one query token per request"
    H = kv_cache.shape[3]
    GROUP = Hq // H
    ND = D // BLOCK
    DATA_BYTES = D // 2
    BLOCK_SIZE = kv_cache.shape[2]
    dev = q.device

    if block_n is None:
        block_n = _DECODE_BLOCK_N
    if split is None:
        env = os.environ.get("VLLM_NVFP4_DECODE_SPLIT")
        if env is not None:
            split = int(env)
        else:
            target = int(_sm_count(dev.index) * _DECODE_WAVES)
            split = max(1, min(_DECODE_MAX_SPLIT, -(-target // (B * H))))
    GPAD = 1 << max(0, (GROUP - 1)).bit_length()
    GPAD = max(GPAD, 16)  # tl.dot needs M >= 16

    qc = q.reshape(B, Hq, D).contiguous()
    bt = block_table.to(torch.int32)
    sl = seq_lens.to(torch.int32)

    out = torch.empty((B * Hq, D), dtype=torch.bfloat16, device=dev)
    npart = B * H * GROUP * split
    pm = torch.empty((npart,), dtype=torch.float32, device=dev)
    pl = torch.empty((npart,), dtype=torch.float32, device=dev)
    pacc = torch.empty((npart, D), dtype=torch.float32, device=dev)
    write_final = split == 1

    _paged_decode_kernel[(B * H * split,)](
        qc, kv_cache, bt, sl, pm, pl, pacc, out,
        sm_scale, Hq,
        GROUP=GROUP, GPAD=GPAD, D=D, H=H, ND=ND, DATA_BYTES=DATA_BYTES,
        BLOCK_SIZE=BLOCK_SIZE, SPLIT=split, BLOCK_N=block_n,
        s_cache_blk=kv_cache.stride(0), s_cache_side=kv_cache.stride(1),
        s_cache_tok=kv_cache.stride(2), s_cache_h=kv_cache.stride(3),
        s_bt=bt.stride(0), WRITE_FINAL=write_final,
        num_warps=_DECODE_NUM_WARPS, num_stages=_DECODE_NUM_STAGES,
    )
    if not write_final:
        _decode_combine_kernel[(B * Hq,)](pm, pl, pacc, out, D=D, SPLIT=split)
    return out.reshape(N, Hq, D)


def nvfp4_kivi_gather_dequant(
    kv_cache: torch.Tensor,  # uint8 [num_blocks, 2, block_size, H, full_dim]
    block_table: torch.Tensor,  # int32 [B, max_blocks]
    seq_lens: torch.Tensor,  # int32 [B]
    head_size: int,
    num_kv_heads: int,
    max_seq: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequant the cached NVFP4 K and V back to bf16 dense tensors
    [B, H, max_seq, D] for use with flash/SDPA attention."""
    B = seq_lens.shape[0]
    H = num_kv_heads
    D = head_size
    ND = D // BLOCK
    DATA_BYTES = D // 2
    FULL_DIM = DATA_BYTES + ND
    BLOCK_SIZE = kv_cache.shape[2]
    device = kv_cache.device

    k_out = torch.zeros((B, H, max_seq, D), dtype=torch.bfloat16, device=device)
    v_out = torch.zeros((B, H, max_seq, D), dtype=torch.bfloat16, device=device)
    grid = (max_seq, B, H)
    for side, out in ((0, k_out), (1, v_out)):
        _gather_dequant_kernel[grid](
            kv_cache,
            block_table,
            seq_lens,
            out,
            B,
            H=H,
            D=D,
            ND=ND,
            DATA_BYTES=DATA_BYTES,
            FULL_DIM=FULL_DIM,
            KV_SIDE=side,
            BLOCK_SIZE=BLOCK_SIZE,
            MAX_SEQ=max_seq,
            s_bt=block_table.stride(0),
            s_cache_blk=kv_cache.stride(0),
            s_cache_side=kv_cache.stride(1),
            s_cache_tok=kv_cache.stride(2),
            s_cache_h=kv_cache.stride(3),
            s_out_b=out.stride(0),
            s_out_h=out.stride(1),
            s_out_s=out.stride(2),
        )
    return k_out, v_out
