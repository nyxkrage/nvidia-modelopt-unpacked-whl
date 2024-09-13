# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Module containing native torch layer-specific symbolic information and implementations."""

import re
from typing import List, Optional, Set

from torch import nn

from ..symbols import Symbol, SymInfo, SymMap

__all__ = ["SymDepth"]


def _get_ndim(mod: nn.Module) -> Optional[int]:
    """Get ndim of a module."""
    ndim_matches = re.findall(r"\d[dD]", type(mod).__name__)
    assert len(ndim_matches) < 2, f"Found multiple ndim matches in {type(mod).__name__}"
    return int(ndim_matches[0][0]) if ndim_matches else None


@SymMap.register(nn.Linear)
def get_linear_sym_info(mod: nn.Module) -> SymInfo:
    in_features = Symbol(cl_type=Symbol.CLType.INCOMING, elastic_dims={-1})
    out_features = Symbol(is_searchable=True, cl_type=Symbol.CLType.OUTGOING, elastic_dims={-1})
    return SymInfo(in_features=in_features, out_features=out_features)


@SymMap.register(
    [
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
    ]
)
def get_norm_sym_info(mod: nn.Module) -> SymInfo:
    # extract elastic dims based on norm type
    ndim = _get_ndim(mod)
    if ndim == 1 or isinstance(mod, nn.SyncBatchNorm):
        edims = {1}
    else:
        edims = {1, -(ndim + 1)}

    num_features = Symbol(cl_type=Symbol.CLType.INCOMING, elastic_dims=edims)
    return SymInfo(is_shape_preserving=True, num_features=num_features)


@SymMap.register(nn.LayerNorm)
def get_layer_norm_sym_info(mod: nn.Module) -> SymInfo:
    num_features = Symbol(cl_type=Symbol.CLType.INCOMING, elastic_dims={-1})
    return SymInfo(is_shape_preserving=True, num_features=num_features)


@SymMap.register(nn.GroupNorm)
def get_groupnorm_sym_info(mod: nn.Module) -> SymInfo:
    num_channels = Symbol(is_sortable=False, cl_type=Symbol.CLType.INCOMING, elastic_dims={1})
    return SymInfo(is_shape_preserving=True, num_channels=num_channels)


class SymDepth(Symbol):
    """A symbolic parameter representing depth."""

    _extra_repr_attrs = ["min_depth", "max_depth"]

    def __init__(self, *args, max_depth, **kwargs):
        """Constructor."""
        super().__init__(*args, **kwargs)
        # whether idx is skippable
        self._is_skippable: List[bool] = [False] * max_depth
        self._max_depth = max_depth

    def is_skippable(self, idx: int) -> bool:
        """Return whether idx is skippable."""
        return self._is_skippable[idx]

    def set_skippable(self, idx: int, val: bool) -> None:
        """Set whether idx is skippable."""
        self._is_skippable[idx] = val

    @property
    def skippable_idxs(self) -> List[int]:
        """Return sorted list of skippable idxs."""
        return [i for i in range(self.max_depth) if self._is_skippable[i]]

    @property
    def max_depth(self) -> int:
        """Return max depth."""
        return self._max_depth

    @property
    def min_depth(self) -> int:
        """Return min depth."""
        return self.max_depth - len(self.skippable_idxs)

    def disable(self, _memo: Optional[Set["Symbol"]] = None) -> None:
        """Disable symbol."""
        super().disable(_memo)
        for i in range(len(self._is_skippable)):
            self._is_skippable[i] = False

    def link_to(self, sp_parent: Symbol) -> None:
        """Link to another symbol."""
        raise RuntimeError("SymDepth cannot be linked to another symbol!")


@SymMap.register(nn.Sequential, is_explicit_leaf=False)
def get_sequential_sym_info(module: nn.Sequential) -> SymInfo:
    return SymInfo(depth=SymDepth(max_depth=len(module), is_searchable=True))


@SymMap.register(
    [nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]
)
def get_conv_sym_info(mod: nn.Module) -> SymInfo:
    # let's register kernel size first (we can always do that)
    symbols = {"kernel_size": Symbol(is_searchable=True)}

    # out/in channels are registered as outgoing/incoming symbols
    ndim = _get_ndim(mod) or -2  # for ndim==None elastic_dims will be {1,-(-2+1)} == {1}!
    out_channels = Symbol(
        is_searchable=True, cl_type=Symbol.CLType.OUTGOING, elastic_dims={1, -(ndim + 1)}
    )
    in_channels = Symbol(cl_type=Symbol.CLType.INCOMING, elastic_dims={1, -(ndim + 1)})

    symbols["out_channels"] = out_channels
    symbols["in_channels"] = in_channels

    # we will disable cross-layer symbols here since we don't handle this case currently
    if mod.groups > 1 and mod.in_channels != mod.out_channels:
        in_channels.disable()
        out_channels.disable()
    # for groups > 1, we treat it as pass-through and link out_channels to in_channels
    elif mod.groups > 1:
        out_channels.link_to(in_channels)
        # we can only sort if it's depthwise (groups == in_channels == out_channels)
        if mod.groups != mod.in_channels:
            in_channels.is_sortable = False
            out_channels.is_sortable = False

    return SymInfo(is_shape_preserving=False, **symbols)
