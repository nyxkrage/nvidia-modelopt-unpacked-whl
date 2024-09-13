# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Utilities to describe symbols in the dynamic attention module."""

from torch import nn
from transformers.models.bert.modeling_bert import BertAttention
from transformers.models.gptj.modeling_gptj import GPTJAttention

from ..symbols import Symbol, SymInfo, SymMap

__all__ = ["SymAttentionHead"]


class SymAttentionHead(Symbol):
    """Just a special class to mark the attention head symbol."""


def get_hf_attn_sym_info(sortable_attn: bool = False) -> SymInfo:
    # embed_dim is registered as elastic incoming symbol (we don't support sorting for now!)
    embed_dim = Symbol(is_sortable=False, cl_type=Symbol.CLType.INCOMING, elastic_dims={-1})

    # num_attention_heads is registered as a special symbol
    num_attention_heads = SymAttentionHead(is_sortable=sortable_attn, is_searchable=True)

    # hidden_dim is linked to num_attention_heads. Correct handling of dependencies done in hps
    # NOTE: we assume hidden_dim is 1st dependency of num_attention_heads in hps!
    hidden_dim = Symbol(is_sortable=sortable_attn, elastic_dims={-1})
    hidden_dim.link_to(num_attention_heads)

    return SymInfo(
        is_shape_preserving=True,
        num_attention_heads=num_attention_heads,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
    )


@SymMap.register([BertAttention])
def get_hf_attn_sym_info_sortable(mod: nn.Module) -> SymInfo:
    return get_hf_attn_sym_info(sortable_attn=True)


@SymMap.register([GPTJAttention])
def get_hf_attn_sym_info_unsortable(mod: nn.Module) -> SymInfo:
    return get_hf_attn_sym_info(sortable_attn=True)
