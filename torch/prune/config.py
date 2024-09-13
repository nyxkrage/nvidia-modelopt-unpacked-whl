# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Default configurations for prune modes."""

from typing import Type

from pydantic import create_model

from modelopt.torch.nas.registry import DMRegistry
from modelopt.torch.opt.config import ModeloptBaseConfig, get_kwargs_for_create_model_with_rules


def _conv_config():
    return {
        "channels_ratio": tuple(0.05 * i for i in range(1, 21)),
        "kernel_size": (),
        "channel_divisor": 32,
    }


def _norm_lin_config():
    return {
        "features_ratio": tuple(0.05 * i for i in range(1, 21)),
        "feature_divisor": 32,
    }


def _get_fastnas_default_rules():
    return {
        "nn.Conv1d": _conv_config(),
        "nn.Conv2d": _conv_config(),
        "nn.Conv3d": _conv_config(),
        "nn.ConvTranspose1d": _conv_config(),
        "nn.ConvTranspose2d": _conv_config(),
        "nn.ConvTranspose3d": _conv_config(),
        "nn.Linear": _norm_lin_config(),
        "nn.BatchNorm1d": _norm_lin_config(),
        "nn.BatchNorm2d": _norm_lin_config(),
        "nn.BatchNorm3d": _norm_lin_config(),
        "nn.SyncBatchNorm": _norm_lin_config(),
        "nn.InstanceNorm1d": _norm_lin_config(),
        "nn.InstanceNorm2d": _norm_lin_config(),
        "nn.InstanceNorm3d": _norm_lin_config(),
        "nn.LayerNorm": _norm_lin_config(),
        "nn.GroupNorm": {k: v for k, v in _conv_config().items() if k != "kernel_size"},
    }


FastNASConfig: Type[ModeloptBaseConfig] = create_model(
    "FastNASConfig",
    **get_kwargs_for_create_model_with_rules(
        registry=DMRegistry,
        default_rules=_get_fastnas_default_rules(),
        doc='Configuration for the ``"fastnas"`` mode.',
    ),
)


GradNASConfig: Type[ModeloptBaseConfig] = create_model(
    "GradNASConfig",
    **get_kwargs_for_create_model_with_rules(
        registry=DMRegistry,
        default_rules=_get_fastnas_default_rules(),
        doc='Configuration for the ``"gradnas"`` mode.',
    ),
)

MCoreGPTMinitronConfig: Type[ModeloptBaseConfig] = create_model(
    "MCoreGPTMinitronConfig",
    **get_kwargs_for_create_model_with_rules(
        registry=DMRegistry,
        default_rules={},  # Dynamically generated rules if Megatron-core is available
        doc='Configuration for the ``"mcore_gpt_minitron"`` mode.',
    ),
)
