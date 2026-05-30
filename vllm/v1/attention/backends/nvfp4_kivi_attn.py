# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4-KIVI software KV-cache attention backend for vLLM.

Sibling of ``int4_kivi_attn.py``: identical control flow and paged cache, but K/V
are stored as nibble-packed **NVFP4 (E2M1)** instead of symmetric INT4.  Same
KIVI layout (per-channel K for full 16-token blocks, per-token V), same fp8_e4m3
per-16 block scales, same byte budget (head_size//2 + head_size//16).  See
``triton_nvfp4_kivi`` for the format rationale and the int4-vs-nvfp4 A/B.

Cache shape (uint8): (num_blocks, 2, block_size, num_kv_heads, full_dim)
  dim 1: 0 = K, 1 = V
  full_dim = head_size//2 (E2M1 data) + head_size//16 (fp8 scales)
"""

import os
from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn.functional as F

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.utils.torch_utils import int4_kivi_kv_cache_full_dim
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import (
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.triton_nvfp4_kivi import (
    nvfp4_kivi_gather_dequant,
    nvfp4_kivi_paged_decode,
    nvfp4_kivi_store,
)

# Fused paged decode is the fast path; set VLLM_NVFP4_NO_FUSED_DECODE=1 to force
# the dense gather+flash fallback (used to A/B correctness and speed).
_USE_FUSED_DECODE = os.environ.get("VLLM_NVFP4_NO_FUSED_DECODE") != "1"

logger = init_logger(__name__)

_HAS_FLASH_ATTN = is_flash_attn_varlen_func_available()
if _HAS_FLASH_ATTN:
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func


class Nvfp4KiviAttentionBackend(AttentionBackend):
    """Attention backend using software NVFP4-KIVI KV-cache compression."""

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["nvfp4_kivi"]

    @staticmethod
    def get_name() -> str:
        return "NVFP4_KIVI"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [16, 32, 64, 128]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        return False

    @staticmethod
    def get_impl_cls() -> type["Nvfp4KiviAttentionImpl"]:
        return Nvfp4KiviAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["Nvfp4KiviMetadataBuilder"]:
        return Nvfp4KiviMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "nvfp4_kivi",
    ) -> tuple[int, ...]:
        # Byte-budget identical to int4_kivi (4-bit data + fp8 per-16 scales).
        full_dim = int4_kivi_kv_cache_full_dim(head_size)
        return (num_blocks, 2, block_size, num_kv_heads, full_dim)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        return kv_cache_dtype == "nvfp4_kivi"

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size > 0 and head_size % 16 == 0


@dataclass
class Nvfp4KiviMetadata(AttentionMetadata):
    seq_lens: torch.Tensor  # (num_reqs,) total context length per request
    slot_mapping: torch.Tensor  # (num_tokens,)
    block_table: torch.Tensor  # (num_reqs, max_num_blocks)
    query_start_loc: torch.Tensor  # (num_reqs + 1,)
    num_actual_tokens: int = 0
    max_query_len: int = 0
    max_seq_len: int = 0
    is_prefill: bool = False
    num_decodes: int = 0
    num_decode_tokens: int = 0
    query_start_loc_cpu: torch.Tensor | None = None
    seq_lens_cpu: torch.Tensor | None = None


class Nvfp4KiviMetadataBuilder(AttentionMetadataBuilder[Nvfp4KiviMetadata]):
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
        return Nvfp4KiviMetadata(
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


class Nvfp4KiviAttentionImpl(AttentionImpl["Nvfp4KiviMetadata"]):
    supports_quant_query_input: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "nvfp4_kivi",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
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

    # ------------------------------------------------------------------ #
    #  Store: quantize new K/V to NVFP4 and write to paged cache slots     #
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
        nvfp4_kivi_store(k, v, kv_cache, slot_mapping, self.head_size)

    def _flash_varlen(self, q, k, v, cu_q, cu_k, max_q, max_k, window=None):
        ws = (-1, -1) if window is None else (window - 1, 0)
        kwargs = dict(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
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
        attn_metadata: "Nvfp4KiviMetadata",
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
            # First-chunk prefill: raw bf16 K/V complete, run flash directly.
            k = key[:N].view(N, self.num_kv_heads, self.head_size)
            v = value[:N].view(N, self.num_kv_heads, self.head_size)
            attn_out = self._flash_varlen(
                q,
                k,
                v,
                attn_metadata.query_start_loc,
                attn_metadata.query_start_loc,
                attn_metadata.max_query_len,
                attn_metadata.max_query_len,
                window=self.sliding_window,
            )
        elif (
            _USE_FUSED_DECODE
            and self.sliding_window is None
            and attn_metadata.max_query_len == 1
        ):
            # Pure-decode batch: fused flash-decode straight off the NVFP4 cache.
            attn_out = self._fused_decode(q, kv_cache, attn_metadata)
        else:
            # General path: dequant the cached NVFP4 K/V and run flash_attn.
            attn_out = self._dequant_and_attend(q, kv_cache, attn_metadata)

        if output.ndim == 3:
            output[:N] = attn_out.to(output.dtype)
        else:
            output[:N] = attn_out.reshape(N, -1).to(output.dtype)
        return output

    def _fused_decode(
        self,
        q: torch.Tensor,  # (N, Hq, D), N == num requests (1 query token each)
        kv_cache: torch.Tensor,
        attn_metadata: "Nvfp4KiviMetadata",
    ) -> torch.Tensor:
        seq_lens = attn_metadata.seq_lens.to(torch.int32)
        block_table = attn_metadata.block_table.to(torch.int32)
        return nvfp4_kivi_paged_decode(
            q, kv_cache, block_table, seq_lens, self.scale
        )

    def _dequant_and_attend(
        self,
        q: torch.Tensor,  # (N, Hq, D)
        kv_cache: torch.Tensor,
        attn_metadata: "Nvfp4KiviMetadata",
    ) -> torch.Tensor:
        N, Hq, D = q.shape
        Hk = self.num_kv_heads
        max_seq = attn_metadata.max_seq_len

        seq_lens = attn_metadata.seq_lens.to(torch.int32)
        block_table = attn_metadata.block_table.to(torch.int32)

        k_dense, v_dense = nvfp4_kivi_gather_dequant(
            kv_cache, block_table, seq_lens, D, Hk, max_seq
        )

        if _HAS_FLASH_ATTN:
            return self._varlen_from_dense(
                q, k_dense, v_dense, attn_metadata, max_seq
            )
        return self._sdpa_from_dense(q, k_dense, v_dense, attn_metadata, max_seq)

    def _varlen_from_dense(self, q, k_dense, v_dense, attn_metadata, max_seq):
        N, Hq, D = q.shape
        Hk = self.num_kv_heads
        device = q.device

        seq_lens_list = (
            attn_metadata.seq_lens_cpu.tolist()
            if attn_metadata.seq_lens_cpu is not None
            else attn_metadata.seq_lens.tolist()
        )
        B = len(seq_lens_list)

        total_k = int(sum(seq_lens_list[:B]))
        k_pack = torch.empty(total_k, Hk, D, dtype=q.dtype, device=device)
        v_pack = torch.empty(total_k, Hk, D, dtype=q.dtype, device=device)
        cu_k = torch.zeros(B + 1, dtype=torch.int32, device=device)
        off = 0
        for i in range(B):
            L = int(seq_lens_list[i])
            k_pack[off : off + L] = k_dense[i, :, :L, :].transpose(0, 1)
            v_pack[off : off + L] = v_dense[i, :, :L, :].transpose(0, 1)
            off += L
            cu_k[i + 1] = off

        return self._flash_varlen(
            q,
            k_pack,
            v_pack,
            attn_metadata.query_start_loc,
            cu_k,
            attn_metadata.max_query_len,
            max_seq,
            window=self.sliding_window,
        )

    def _sdpa_from_dense(self, q, k_dense, v_dense, attn_metadata, max_seq):
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
