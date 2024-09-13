# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Dynamic linear implementations based on torch.nn.modules.linear."""

from typing import Optional, Tuple

import torch
from torch import nn

from modelopt.torch.opt.dynamic import DynamicModule
from modelopt.torch.utils import make_divisible

from ..registry import DMRegistry
from ..traced_hp import TracedHp
from .utils import get_sliced_tensor

__all__ = ["_DynamicLinear"]


@DMRegistry.register({nn.Linear: "nn.Linear"})
class _DynamicLinear(DynamicModule):
    """An ``nn.Linear`` layer with dynamic hyperparams."""

    @staticmethod
    def _get_weight(mod: "_DynamicLinear", weight: torch.Tensor) -> torch.Tensor:
        return get_sliced_tensor(mod, weight, "out_features", "in_features")

    @staticmethod
    def _get_bias(mod: "_DynamicLinear", bias: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return get_sliced_tensor(mod, bias, "out_features")

    def _estimate_importance(self) -> TracedHp.Importance:
        return self._parameters["weight"].detach().norm(dim=0)

    def _setup(self):
        # register hyperparameters
        self._register_hparam("in_features", TracedHp(list(range(1, self.in_features + 1))))
        self._register_hparam("out_features", TracedHp(list(range(1, self.out_features + 1))))

        # register dynamic attributes of the class
        self._register_dynamic_attribute("weight", self._get_weight)
        self._register_dynamic_attribute("bias", self._get_bias)

        # register importance for in_features
        self.get_hparam("in_features").register_importance(self._estimate_importance)

    def modify(
        self, *, features_ratio: Optional[Tuple[float, ...]] = None, feature_divisor: int = 1
    ):
        """Modify the dynamic choices of the module according to provided keyword arguments.

        Args:
            features_ratio: The ratios of the desired number of output/input features over original
                number of output/input features.
            feature_divisor: The divisor of the output/input features.
        """
        # modify both in_features and out_features
        features = ["in_features", "out_features"]
        for feature in features:
            hp = self.get_hparam(feature)
            if features_ratio is not None:
                choices = {r * hp.original for r in features_ratio}
            else:
                choices = set(hp.choices)
            choices = {int(make_divisible(c, feature_divisor)) for c in choices}
            hp.choices = list(set(hp.choices) & choices | {hp.original})
