# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared base classes for KIVI-style quantized KV-cache attention backends.

Both ``int4_kivi_attn`` (symmetric INT4) and ``nvfp4_kivi_attn`` (E2M1 NVFP4)
use identical paged-cache layouts, metadata, and control flow.  Only the
codec-specific store / gather / decode kernels differ.  This module factors
out everything that's shared, so each backend supplies only its codec callables.

Usage::

    from vllm.v1.attention.backends._kivi_attn_base import (
        KiviAttentionImplBase,
        KiviMetadata,
        KiviMetadataBuilder,
    )

    class MyKiviAttentionImpl(KiviAttentionImplBase):
        def _store(self, key, value, kv_cache, slot_mapping, head_size):
            my_codec_store(key, value, kv_cache, slot_mapping, head_size)

        def _gather_dequant(self, kv_cache, block_table, seq_lens, head_size, num_kv_heads, max_seq):
            return my_codec_gather_dequant(kv_cache, block_table, seq_lens, head_size, num_kv_heads, max_seq)

        def _fused_decode_kernel(self, q, kv_cache, block_table, seq_lens, scale):
            return my_codec_paged_decode(q, kv_cache, block_table, seq_lens, scale)
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn.functional as F

from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
)
from vllm.v1.attention.backends.fa_utils import (
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills

_HAS_FLASH_ATTN = is_flash_attn_varlen_func_available()
if _HAS_FLASH_ATTN:
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func


@dataclass
class KiviMetadata(AttentionMetadata):
    """Shared attention metadata for all KIVI backends."""
    seq_lens: torch.Tensor          # (num_reqs,) total context length per request
    slot_mapping: torch.Tensor      # (num_tokens,)
    block_table: torch.Tensor       # (num_reqs, max_num_blocks)
    query_start_loc: torch.Tensor   # (num_reqs + 1,)
    num_actual_tokens: int = 0
    max_query_len: int = 0
    max_seq_len: int = 0
    is_prefill: bool = False
    num_decodes: int = 0
    num_decode_tokens: int = 0
    query_start_loc_cpu: torch.Tensor | None = None
    seq_lens_cpu: torch.Tensor | None = None


class KiviMetadataBuilder(AttentionMetadataBuilder[KiviMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        cam = common_attn_metadata
        assert self.reorder_batch_threshold is not None
        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            cam, decode_threshold=self.reorder_batch_threshold
        )
        return KiviMetadata(
            seq_lens=cam.seq_lens,
            slot_mapping=cam.slot_mapping,
            block_table=cam.block_table_tensor,
            query_start_loc=cam.query_start_loc,
            num_actual_tokens=cam.num_actual_tokens,
            max_query_len=cam.max_query_len,
            max_seq_len=cam.max_seq_len,
            is_prefill=(cam.max_query_len > 1),
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            query_start_loc_cpu=cam.query_start_loc_cpu,
            seq_lens_cpu=cam.seq_lens_cpu_upper_bound,
        )


class KiviAttentionImplBase(AttentionImpl[KiviMetadata], abc.ABC):
    """Codec-agnostic KIVI attention implementation.

    Subclasses implement :meth:`_store`, :meth:`_gather_dequant`, and
    :meth:`_fused_decode_kernel` for their specific 4-bit codec.
    """

    supports_quant_query_input: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        use_fused_decode: bool = True,
        **kwargs,
    ):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = sliding_window
        self.fa_version = get_flash_attn_version(head_size=head_size)
        self._use_fused_decode = use_fused_decode

    # ------------------------------------------------------------------ #
    # Codec interface — implement in each backend subclass.               #
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def _store(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        head_size: int,
    ) -> None:
        """Quantize and store new K/V tokens into the paged cache."""

    @abc.abstractmethod
    def _gather_dequant(
        self,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        head_size: int,
        num_kv_heads: int,
        max_seq: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantize all cached K and V back to bf16 dense tensors [B,H,S,D]."""

    @abc.abstractmethod
    def _fused_decode_kernel(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Fused flash-decode directly off the packed quantised cache."""

    # ------------------------------------------------------------------ #
    # Shared control flow.                                                #
    # ------------------------------------------------------------------ #

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        N = slot_mapping.shape[0]
        if N <= 0:
            return
        k = key[:N].view(N, self.num_kv_heads, self.head_size)
        v = value[:N].view(N, self.num_kv_heads, self.head_size)
        self._store(k, v, kv_cache, slot_mapping, self.head_size)

    def _flash_varlen(self, q, k, v, cu_q, cu_k, max_q, max_k, window=None):
        ws = (-1, -1) if window is None else (window - 1, 0)
        kwargs = dict(
            q=q, k=k, v=v,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=max_q, max_seqlen_k=max_k,
            softmax_scale=self.scale,
            causal=True,
            window_size=ws,
        )
        if self.fa_version is not None:
            kwargs["fa_version"] = self.fa_version
        return flash_attn_varlen_func(**kwargs)

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: KiviMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = query.shape[0]
        if output is None:
            output = torch.zeros(
                num_tokens,
                self.num_heads * self.head_size,
                dtype=query.dtype,
                device=query.device,
            )
        if attn_metadata is None:
            return output.fill_(0)

        N = attn_metadata.num_actual_tokens
        if N <= 0:
            return output.fill_(0)

        q = query[:N].view(N, self.num_heads, self.head_size)

        if (
            _HAS_FLASH_ATTN
            and attn_metadata.num_decodes == 0
            and attn_metadata.max_query_len == attn_metadata.max_seq_len
            and attn_metadata.max_query_len > 1
        ):
            k = key[:N].view(N, self.num_kv_heads, self.head_size)
            v = value[:N].view(N, self.num_kv_heads, self.head_size)
            attn_out = self._flash_varlen(
                q, k, v,
                attn_metadata.query_start_loc, attn_metadata.query_start_loc,
                attn_metadata.max_query_len, attn_metadata.max_query_len,
                window=self.sliding_window,
            )
        elif (
            self._use_fused_decode
            and self.sliding_window is None
            and attn_metadata.max_query_len == 1
        ):
            seq_lens = attn_metadata.seq_lens.to(torch.int32)
            block_table = attn_metadata.block_table.to(torch.int32)
            attn_out = self._fused_decode_kernel(q, kv_cache, block_table, seq_lens, self.scale)
        else:
            attn_out = self._dequant_and_attend(q, kv_cache, attn_metadata)

        if output.ndim == 3:
            output[:N] = attn_out.to(output.dtype)
        else:
            output[:N] = attn_out.reshape(N, -1).to(output.dtype)
        return output

    def _dequant_and_attend(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: KiviMetadata,
    ) -> torch.Tensor:
        _, Hq, D = q.shape
        max_seq = attn_metadata.max_seq_len
        seq_lens = attn_metadata.seq_lens.to(torch.int32)
        block_table = attn_metadata.block_table.to(torch.int32)
        k_dense, v_dense = self._gather_dequant(
            kv_cache, block_table, seq_lens, D, self.num_kv_heads, max_seq
        )
        if _HAS_FLASH_ATTN:
            return self._varlen_from_dense(q, k_dense, v_dense, attn_metadata, max_seq)
        return self._sdpa_from_dense(q, k_dense, v_dense, attn_metadata, max_seq)

    def _varlen_from_dense(self, q, k_dense, v_dense, attn_metadata, max_seq):
        """Pack K/V into a varlen-contiguous buffer and run flash_attn.

        Uses vectorised boolean indexing (no Python loop over batch dimension)
        to build the packed buffers from the dense [B, H, max_seq, D] tensors.
        """
        _, Hq, D = q.shape
        Hk = self.num_kv_heads
        device = q.device

        seq_lens = attn_metadata.seq_lens.to(torch.int32)
        B = seq_lens.shape[0]

        # cu_k: [B+1] cumulative sequence lengths.
        cu_k = torch.zeros(B + 1, dtype=torch.int32, device=device)
        torch.cumsum(seq_lens[:B], dim=0, out=cu_k[1:])

        # Boolean mask [B, max_seq] selecting valid (non-padding) positions.
        pos = torch.arange(max_seq, device=device).unsqueeze(0)  # [1, max_seq]
        mask = pos < seq_lens[:B].unsqueeze(1).to(device)         # [B, max_seq]

        # k_dense/v_dense: [B, H, max_seq, D] -> [B, max_seq, H, D] -> [total_k, H, D]
        k_pack = k_dense.permute(0, 2, 1, 3)[mask]  # [total_k, H, D]
        v_pack = v_dense.permute(0, 2, 1, 3)[mask]  # [total_k, H, D]

        return self._flash_varlen(
            q, k_pack, v_pack,
            attn_metadata.query_start_loc, cu_k,
            attn_metadata.max_query_len, max_seq,
            window=self.sliding_window,
        )

    def _sdpa_from_dense(self, q, k_dense, v_dense, attn_metadata, max_seq):
        """SDPA fallback (no flash_attn). Per-request causal attention."""
        N, Hq, D = q.shape
        Hk = self.num_kv_heads
        device = q.device
        use_gqa = Hk < Hq
        qsl = (
            attn_metadata.query_start_loc_cpu.tolist()
            if attn_metadata.query_start_loc_cpu is not None
            else attn_metadata.query_start_loc.tolist()
        )
        seq_lens_list = (
            attn_metadata.seq_lens_cpu.tolist()
            if attn_metadata.seq_lens_cpu is not None
            else attn_metadata.seq_lens.tolist()
        )
        out = torch.empty(N, Hq, D, dtype=q.dtype, device=device)
        for i in range(len(seq_lens_list)):
            qs, qe = qsl[i], qsl[i + 1]
            q_len = qe - qs
            if q_len <= 0:
                continue
            seq_len = int(seq_lens_list[i])
            cached = seq_len - q_len
            q_t = q[qs:qe].transpose(0, 1).unsqueeze(0)
            k_t = k_dense[i, :, :seq_len, :].unsqueeze(0)
            v_t = v_dense[i, :, :seq_len, :].unsqueeze(0)
            q_pos = torch.arange(q_len, device=device).unsqueeze(1) + cached
            k_pos = torch.arange(seq_len, device=device).unsqueeze(0)
            mask = k_pos <= q_pos
            if self.sliding_window is not None:
                mask = mask & (k_pos > q_pos - self.sliding_window)
            o = F.scaled_dot_product_attention(
                q_t, k_t, v_t, attn_mask=mask, scale=self.scale, enable_gqa=use_gqa
            )
            out[qs:qe] = o[0].transpose(0, 1).to(q.dtype)
        return out
