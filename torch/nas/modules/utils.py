# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Internal module for utility functions."""
from typing import List, Optional

import torch

from modelopt.torch.opt.dynamic import DynamicModule
from modelopt.torch.opt.hparam import Hparam


def get_sliced_tensor_by_slices(
    tensor: Optional[torch.Tensor],
    slices: List[Hparam.ActiveSlice],
) -> Optional[torch.Tensor]:
    """Get the tensor based on the active slice."""
    if tensor is None:
        return tensor

    # check if we can return the original tensor
    if all(
        isinstance(s, slice) and (s.stop is None or s.indices(s.stop) == (0, tensor.shape[i], 1))
        for i, s in enumerate(slices)
    ):
        return tensor

    # slice tensor with minimal number of index operations
    tensor_sliced = tensor
    for i, _ in enumerate(slices):
        if sum(not isinstance(s, slice) for s in slices) < 2:
            tensor_sliced = tensor_sliced[slices]
            break
        tensor_sliced = tensor_sliced[slices[: i + 1]]
        slices[i] = slice(None)  # replace with a vanilla slice ("[:]") for next slicing iteration

    # return sliced, contiguous tensor
    return tensor_sliced.contiguous()


def get_sliced_tensor(
    mod: DynamicModule,
    tensor: Optional[torch.Tensor],
    *hp_names: str,
) -> Optional[torch.Tensor]:
    """Get the tensor based on the slices."""
    slices = [mod.get_hparam(hp_name).active_slice for hp_name in hp_names]
    return get_sliced_tensor_by_slices(tensor, slices)
