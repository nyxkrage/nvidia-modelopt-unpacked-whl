# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Plugins for tracing Megatron modules."""

import operator
from typing import List, Union

from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.mlp import MLP
from torch.fx import Node

from modelopt.torch.utils.graph import NodeTarget

from ..analyzer import GraphDependencyProcessor, NodeProcessor
from ..modules.nn import get_layer_norm_sym_info, get_linear_sym_info
from ..symbols import Symbol, SymInfo, SymMap


@SymMap.register([ColumnParallelLinear, RowParallelLinear])
def get_megatron_parallel_linear_sym_info(
    mod: Union[ColumnParallelLinear, RowParallelLinear],
) -> SymInfo:
    """Get symbol information for ``ColumnParallelLinear`` and ``RowParallelLinear`` layers."""
    return get_linear_sym_info(mod)


@SymMap.register(MLP)
def get_megatron_mlp_sym_info(mod: MLP) -> SymInfo:
    """Get symbol information for ``MLP`` layer."""
    ffn_hidden_size = Symbol(is_searchable=True, elastic_dims={-1})
    return SymInfo(is_shape_preserving=True, ffn_hidden_size=ffn_hidden_size)


@SymMap.register(FusedLayerNorm)
def get_megatron_fused_ln_sym_info(mod: FusedLayerNorm) -> SymInfo:
    """Get symbol information for ``FusedLayerNorm`` layers."""
    return get_layer_norm_sym_info(mod)


@SymMap.register(SelfAttention)
def get_megatron_self_attention_sym_info(mod: SelfAttention) -> SymInfo:
    """Get symbol information for ``SelfAttention`` layers."""
    num_heads_per_group = Symbol(is_sortable=True, is_searchable=True)
    num_query_groups = Symbol(is_sortable=True, is_searchable=True)
    return SymInfo(
        is_shape_preserving=True,
        num_heads_per_group=num_heads_per_group,
        num_query_groups=num_query_groups,
    )


@GraphDependencyProcessor.register_node_processor
class MegatronGetattrPassthroughNodeProcessor(NodeProcessor):
    """Node for handling getattr on MegatronModule outputs.

    For RowParallelLinear and ColumnParallelLinear, its output is a tuple so we need to handle the
    getattr on the output as a passthrough operation to not make the Linear layer symbols constant.
    """

    def is_special_node(self, node: Node, target: NodeTarget) -> bool:
        """Return whether node is to be supported by this processor."""
        if (
            target == operator.getitem
            and isinstance(
                self._get_root_target(node.args[0]), (ColumnParallelLinear, RowParallelLinear)
            )
            and isinstance(node.args[1], int)
        ):
            return True
        return False

    def process(self, node: Node, id: int, input_nodes: List[Node]) -> None:
        """Process like a pass-through node."""
        self._process_passthrough(node, id, input_nodes)
