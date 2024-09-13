# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Support quantization for apex linear layers."""


from functools import partial

import apex.transformer.tensor_parallel.layers as apex_parallel
from apex.transformer.parallel_state import get_data_parallel_group, get_tensor_model_parallel_group

from modelopt.torch.quantization.nn.modules.quant_linear import _QuantLinear
from modelopt.torch.utils.distributed import ParallelState

from ..nn import QuantModuleRegistry
from .custom import _ParallelLinear


class _ApexParallelLinear(_ParallelLinear):

    def initialize_parallel_state(self):
        self._parallel_state = ParallelState(
            get_data_parallel_group(), get_tensor_model_parallel_group()
        )

    def _setup(self):
        quantized_linear_fn = partial(
            _QuantLinear.quantized_linear_fn,
            apex_parallel,
            "linear_with_grad_accumulation_and_async_allreduce",
            self,
        )
        self._forward_impl = quantized_linear_fn
        super()._setup()


@QuantModuleRegistry.register({apex_parallel.ColumnParallelLinear: "apex_ColumnParallelLinear"})
class _ApexColumnParallelLinear(_ApexParallelLinear):
    _is_column_parallel = True


@QuantModuleRegistry.register({apex_parallel.RowParallelLinear: "apex_RowParallelLinear"})
class _ApexRowParallelLinear(_ApexParallelLinear):
    _is_row_parallel = True
