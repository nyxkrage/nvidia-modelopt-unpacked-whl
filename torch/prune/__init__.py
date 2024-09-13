# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""A simplified API for :meth:`modelopt.torch.nas<modelopt.torch.nas>` for pruning algorithms.

This module provides a simplified API for pruning that is based on the NAS infrastructure but
simplifies the overall workflow to accommodate for the simpler nature of pruning algorithms.
"""
# nas is a required - so let's check if it's available
import modelopt.torch.nas

from . import config, mode, plugins
from .pruning import *
