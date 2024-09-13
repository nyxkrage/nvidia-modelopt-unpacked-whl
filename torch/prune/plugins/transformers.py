# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Plugins for pruning for Transformers Attention."""

# import nas plugin to check if it is enabled else raises an Exception
from modelopt.torch.nas.plugins.transformers import *  # noqa: F403

from ..config import FastNASConfig, GradNASConfig


def _n_heads_config():
    return {"n_heads_ratio": None, "n_heads_divisor": 1}


FastNASConfig.register_default(
    {
        "hf.BertAttention": _n_heads_config(),
        "hf.GPTJAttention": _n_heads_config(),
    }
)

GradNASConfig.register_default(
    {
        "hf.BertAttention": _n_heads_config(),
        "hf.GPTJAttention": _n_heads_config(),
    }
)
