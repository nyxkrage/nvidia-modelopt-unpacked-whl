# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Dynamic norm implementations based on norm modules in torch.nn.modules."""

from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from modelopt.torch.opt.dynamic import DynamicModule
from modelopt.torch.utils import make_divisible

from ..registry import DMRegistry
from ..traced_hp import TracedHp
from .utils import get_sliced_tensor

__all__ = ["_DynamicBatchNorm", "_DynamicInstanceNorm", "_DynamicLayerNorm", "_DynamicGroupNorm"]


class _DynamicBatchInstance(DynamicModule):
    """Dynamic base class for batch norm and instance norm layers.

    NOTE: Don't use this class for instance checks. Use _DynamicBatchNorm or _DynamicInstanceNorm
    instead!
    """

    @staticmethod
    def _cut_to_active_features(mod: "_DynamicBatchInstance", value: Optional[torch.Tensor]):
        return get_sliced_tensor(mod, value, "num_features")

    def _setup(self):
        # register hyperparameters
        self._register_hparam("num_features", TracedHp(list(range(1, self.num_features + 1))))

        # register dynamic attributes
        dyn_attrs = ["running_mean", "running_var", "weight", "bias"]
        for attr in dyn_attrs:
            self._register_dynamic_attribute(attr, self._cut_to_active_features)

    def modify(
        self, *, features_ratio: Optional[Tuple[float, ...]] = None, feature_divisor: int = 1
    ):
        """Modify the dynamic choices of the module according to provided keyword arguments.

        Args:
            features_ratio: The ratios of the desired number of features over original number of
                features.
            feature_divisor: The divisor of the number of features.
        """
        hp = self.get_hparam("num_features")
        if features_ratio is not None:
            choices = {r * hp.original for r in features_ratio}
        else:
            choices = set(hp.choices)
        choices = {int(make_divisible(c, feature_divisor)) for c in choices}
        hp.choices = list(set(hp.choices) & choices | {hp.original})


@DMRegistry.register(
    {
        nn.BatchNorm1d: "nn.BatchNorm1d",
        nn.BatchNorm2d: "nn.BatchNorm2d",
        nn.BatchNorm3d: "nn.BatchNorm3d",
        nn.SyncBatchNorm: "nn.SyncBatchNorm",
    }
)
class _DynamicBatchNorm(_DynamicBatchInstance):
    """Just syntactic sugar so we have a common base class for batch norm only."""


@DMRegistry.register(
    {
        nn.InstanceNorm1d: "nn.InstanceNorm1d",
        nn.InstanceNorm2d: "nn.InstanceNorm2d",
        nn.InstanceNorm3d: "nn.InstanceNorm3d",
    }
)
class _DynamicInstanceNorm(_DynamicBatchInstance):
    """Just syntactic sugar so we have a common base class for instance norm only."""


@DMRegistry.register({nn.LayerNorm: "nn.LayerNorm"})
class _DynamicLayerNorm(DynamicModule):
    """An ``nn.LayerNorm`` layer with dynamic hyperparams."""

    @staticmethod
    def _get_normalized_shape(
        mod: "_DynamicLayerNorm", value: Sequence[Union[int, TracedHp]]
    ) -> Tuple:
        return tuple(value[:-1]) + (mod.num_features,)

    @staticmethod
    def _cut_to_active_features(
        mod: "_DynamicLayerNorm", value: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if value is None or value.shape[-1] == mod.num_features:
            return value
        nf_slice = mod.get_hparam("num_features").active_slice
        return value[..., nf_slice].contiguous()

    def _setup(self):
        # construct normalized shape with Hparam as last dimension
        normalized_shape = list(self.normalized_shape)
        normalized_shape[-1] = TracedHp(list(range(1, normalized_shape[-1] + 1)))

        # register the hyperparameter with a new name
        self._register_hparam("num_features", normalized_shape[-1])

        # register dynamic attributes
        dyn_attrs = ["weight", "bias"]
        for attr in dyn_attrs:
            self._register_dynamic_attribute(attr, self._cut_to_active_features)

        self._register_dynamic_attribute("normalized_shape", self._get_normalized_shape)

    def modify(
        self, *, features_ratio: Optional[Tuple[float, ...]] = None, feature_divisor: int = 1
    ):
        """Modify the dynamic choices of the module according to provided keyword arguments.

        Args:
            features_ratio: The ratios of the desired number of features over original number of
                features.
            feature_divisor: The divisor of the number of features.
        """
        hp = self.get_hparam("num_features")
        if features_ratio is not None:
            choices = {r * hp.original for r in features_ratio}
        else:
            choices = set(hp.choices)
        choices = {int(make_divisible(c, feature_divisor)) for c in choices}
        hp.choices = list(set(hp.choices) & choices | {hp.original})


@DMRegistry.register({nn.GroupNorm: "nn.GroupNorm"})
class _DynamicGroupNorm(DynamicModule):
    """An ``nn.GroupNorm`` layer with dynamic hyperparams."""

    _group_size: int

    @staticmethod
    def _get_num_groups(mod: "_DynamicGroupNorm", value: int) -> int:
        return mod.num_channels // mod._group_size

    @staticmethod
    def _cut_to_active_channels(mod: "_DynamicGroupNorm", value: Optional[torch.Tensor]):
        return get_sliced_tensor(mod, value, "num_channels")

    def _setup(self):
        # register num_channels as hyperparameter
        group_size = self.num_channels // self.num_groups
        choices = [
            c
            for c in range(self.num_groups, self.num_channels + 1)
            if c % self.num_groups == 0 and c % group_size == 0
        ]
        self._register_hparam("num_channels", TracedHp(choices, original=self.num_channels))

        # register num_groups as a dynamic attribute so group size is same
        self._register_temp_attribute("_group_size", group_size)
        self._register_dynamic_attribute("num_groups", self._get_num_groups)

        # register dynamic attributes
        dyn_attrs = ["weight", "bias"]
        for attr in dyn_attrs:
            self._register_dynamic_attribute(attr, self._cut_to_active_channels)

    def modify(
        self, *, channels_ratio: Optional[Tuple[float, ...]] = None, channel_divisor: int = 1
    ):
        """Modify the dynamic choices of the module according to provided keyword arguments.

        Args:
            channels_ratio: The ratios of the desired number of out/in channels over original
                number of out/in channels.
            channel_divisor: The divisor of the out/in channels.
        """
        hp = self.get_hparam("num_channels")
        choices: List[int]
        if channels_ratio is not None:
            choices = {r * hp.original for r in channels_ratio}
        else:
            choices = set(hp.choices)
        choices = {int(make_divisible(c, channel_divisor)) for c in choices}
        hp.choices = list(set(hp.choices) & choices | {hp.original})
