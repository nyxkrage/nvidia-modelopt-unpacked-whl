# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Medusa conversion/restore utilities."""

from torch import nn

from modelopt.torch.opt.conversion import ModelLikeModule
from modelopt.torch.opt.dynamic import _DMRegistryCls
from modelopt.torch.opt.mode import ConvertReturnType, MetadataDict

from ..config import MedusaConfig

MedusaDMRegistry = _DMRegistryCls(prefix="Medusa")  # global instance for the registry


def convert_to_medusa_model(model: nn.Module, config: MedusaConfig) -> ConvertReturnType:
    """Convert the model to a medusa model as per `config`."""
    # initialize the true module if necessary
    model = model.init_modellike() if isinstance(model, ModelLikeModule) else model

    original_cls = type(model)
    if original_cls not in MedusaDMRegistry:
        for cls in MedusaDMRegistry._registry:
            if issubclass(original_cls, cls):
                MedusaDMRegistry.register({original_cls: "base_model_class"})(MedusaDMRegistry[cls])
                break

    medusa_model = MedusaDMRegistry.convert(model)
    medusa_model.modify(
        medusa_num_heads=config.medusa_num_heads, medusa_num_layers=config.medusa_num_layers
    )

    # no metadata, all specifed via config.
    metadata = {}

    return medusa_model, metadata


def restore_medusa_model(
    model: nn.Module, config: MedusaConfig, metadata: MetadataDict
) -> nn.Module:
    """Function for restoring a previously convert model to a medusa model."""
    # the metadata should be empty
    assert not metadata, "No metadata expected!"

    return convert_to_medusa_model(model, config)[0]
