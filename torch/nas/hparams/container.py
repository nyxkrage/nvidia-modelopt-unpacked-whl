# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Hparam for Depth."""


from typing import Callable, Dict

from modelopt.torch.trace import Symbol, SymDepth

from ..traced_hp import TracedHp, TracedHpRegistry

__all__ = ["DepthHparam"]


@TracedHpRegistry.register(SymDepth)
class DepthHparam(TracedHp):
    """Hparam describing depth."""

    def _resolve_dependencies(
        self, sym: Symbol, get_hp: Callable[[Symbol], TracedHp]
    ) -> Dict[Symbol, TracedHp]:
        assert isinstance(sym, SymDepth), f"Unexpected type {type(sym)} for {sym}!"
        assert not sym._dependencies, "Depth should not have any dependencies!"
        assert not sym._parent, "Depth should not have any parents!"

        # we can only skip layers when all layers after are skippable
        choices = {sym.max_depth}
        for i in range(sym.max_depth - 1, -1, -1):
            if not sym.is_skippable(i):
                break
            choices.add(i)

        # record choices and skippable layers
        self.choices = list(choices & set(self.choices))

        return super()._resolve_dependencies(sym, get_hp)
