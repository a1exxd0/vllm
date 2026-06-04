# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT4-KIVI software KV-cache attention backend.

Thin wrapper over :class:`~vllm.v1.attention.backends._kivi_attn_base.KiviAttentionImplBase`.
All shared control flow, metadata, and the KV-cache write path live in the base
class.  This module supplies only the INT4 codec callables.

See ``triton_int4_kivi.py`` for the quantisation details.
"""

from __future__ import annotations

import os
from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.utils.torch_utils import int4_kivi_kv_cache_full_dim
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends._kivi_attn_base import (
    KiviAttentionImplBase,
    KiviMetadata,
    KiviMetadataBuilder,
)
from vllm.v1.attention.ops.triton_int4_kivi import (
    int4_kivi_gather_dequant,
    int4_kivi_paged_decode,
    int4_kivi_store,
)

_USE_FUSED_DECODE = os.environ.get("VLLM_INT4_NO_FUSED_DECODE") != "1"

logger = init_logger(__name__)

# Re-export for the backend registry.
Int4KiviMetadata = KiviMetadata
Int4KiviMetadataBuilder = KiviMetadataBuilder


class Int4KiviAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["int4_kivi"]

    @staticmethod
    def get_name() -> str:
        return "INT4_KIVI"

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
    def get_impl_cls() -> type[AttentionImpl]:
        return Int4KiviAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[Int4KiviMetadataBuilder]:
        return Int4KiviMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int, block_size: int, num_kv_heads: int, head_size: int,
        cache_dtype_str: str = "int4_kivi",
    ) -> tuple[int, ...]:
        return (num_blocks, 2, block_size, num_kv_heads, int4_kivi_kv_cache_full_dim(head_size))

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        return kv_cache_dtype == "int4_kivi"

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size > 0 and head_size % 16 == 0


class Int4KiviAttentionImpl(KiviAttentionImplBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_fused_decode=_USE_FUSED_DECODE, **kwargs)

    def _store(self, key, value, kv_cache, slot_mapping, head_size):
        int4_kivi_store(key, value, kv_cache, slot_mapping, head_size)

    def _gather_dequant(self, kv_cache, block_table, seq_lens, head_size, num_kv_heads, max_seq):
        return int4_kivi_gather_dequant(kv_cache, block_table, seq_lens, head_size, num_kv_heads, max_seq)

    def _fused_decode_kernel(self, q, kv_cache, block_table, seq_lens, scale):
        return int4_kivi_paged_decode(q, kv_cache, block_table, seq_lens, scale)
