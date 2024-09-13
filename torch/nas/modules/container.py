# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Dynamic sequential implementation (variable depth based on torch.nn.modules.container)."""

from collections import OrderedDict
from typing import Any, Callable

from torch import nn

from modelopt.torch.opt.dynamic import DynamicModule

from ..registry import DMRegistry
from ..traced_hp import TracedHp

__all__ = ["_DynamicSequential"]


def _activate_depth(func: Callable) -> Callable:
    """A decorator for enabling dynamic depth within methods in _DynamicSequential.

    The decorated method (and methods within the decorated method) will see any attributes with
    dynamic depth enabled.
    """

    def func_with_dynamic_depth(self: "_DynamicSequential", *args, **kwargs) -> Any:
        """Call func and return value wrapped with temporarily enabling dynamic depth."""
        val = self._dynamic_depth
        self._dynamic_depth = True
        ret = func(self, *args, **kwargs)
        self._dynamic_depth = val
        return ret

    return func_with_dynamic_depth


@DMRegistry.register({nn.Sequential: "nn.Sequential"})
class _DynamicSequential(DynamicModule):
    """An ``nn.Sequential`` layer with dynamic hyperparams and variable ``depth``."""

    _dynamic_depth: bool

    @_activate_depth
    def forward(self, input):
        """Forward with activated variable depth."""
        return super().forward(input)

    @_activate_depth
    def export(self) -> nn.Module:
        """Export with activated variable depth."""
        return super().export()

    @_activate_depth
    def __repr__(self):
        """__repr__ with dynamic depth enabled -> str(self) will only show active subnet."""
        return super().__repr__()

    @_activate_depth
    def extra_repr(self):
        """extra_repr with dynamic depth enabled -> str(self) will only show active subnet."""
        return super().extra_repr()

    @staticmethod
    def _get_modules(mod: "_DynamicSequential", modules: OrderedDict) -> OrderedDict:
        if mod.depth < len(modules) and mod._dynamic_depth:
            return OrderedDict(list(modules.items())[: mod.depth])
        return modules

    def _setup(self):
        # register temp attribute to keep track of whether dynamic depth is on
        self._register_temp_attribute("_dynamic_depth", False)

        # register hyperparameters
        self._register_hparam("depth", TracedHp(list(range(0, len(self) + 1))))

        # register _modules as a dynamic attribute
        self._register_dynamic_attribute("_modules", self._get_modules)

    def modify(self, *, min_depth: int = 0):
        """Modify the dynamic choices of the module according to provided keyword arguments.

        Args:
            min_depth: The minimum depth of the module.
        """
        hp = self.get_hparam("depth")
        hp.choices = [d for d in hp.choices if d >= min_depth]
