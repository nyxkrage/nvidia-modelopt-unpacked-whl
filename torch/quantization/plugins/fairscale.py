# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Support quantization for megatron linear layers."""

import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import get_data_parallel_group, get_model_parallel_group
from fairscale.nn.model_parallel.layers import ColumnParallelLinear, RowParallelLinear

from modelopt.torch.utils.distributed import ParallelState

from ..nn import QuantModuleRegistry
from .custom import _ParallelLinear

__all__ = []


class _FairscaleParallelLinear(_ParallelLinear):
    _functionals_to_replace = [(F, "linear")]

    def initialize_parallel_state(self):
        self._parallel_state = ParallelState(get_data_parallel_group(), get_model_parallel_group())


@QuantModuleRegistry.register({ColumnParallelLinear: "fairscale_ColumnParallelLinear"})
class _FairscaleColumnParallelLinear(_FairscaleParallelLinear):
    _is_column_parallel = True


@QuantModuleRegistry.register({RowParallelLinear: "fairscale_RowParallelLinear"})
class _FairscaleRowParallelLinear(_FairscaleParallelLinear):
    _is_row_parallel = True
