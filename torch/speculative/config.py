# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Configurations for speculative decoding modes."""


from modelopt.torch.opt.config import ModeloptBaseConfig, ModeloptField


class MedusaConfig(ModeloptBaseConfig):
    """Medusa config."""

    medusa_num_heads: int = ModeloptField(
        default=2,
        description=("The number of medusa heads added to the model."),
    )

    medusa_num_layers: int = ModeloptField(
        default=1,
        description=("The number of ResBlocks used in medusa head."),
    )
