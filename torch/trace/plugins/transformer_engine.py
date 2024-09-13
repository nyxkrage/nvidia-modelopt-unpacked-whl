# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Plugins for tracing Transformer Engine modules."""

import torch.nn as nn
import transformer_engine as te

from ..modules.nn import get_layer_norm_sym_info
from ..symbols import SymInfo, SymMap


@SymMap.register([te.pytorch.LayerNorm, te.pytorch.RMSNorm])
def get_te_norm_sym_info(mod: nn.Module) -> SymInfo:
    """Get symbol information for ``FusedLayerNorm`` layers."""
    return get_layer_norm_sym_info(mod)
