# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Main module for NAS/Pruning-based model design and optimization."""
# isort: off
# Import dynamic modules first to avoid KeyError: 'nn.Conv1d is not registered for a dynamic module!'
from . import modules

# isort: on

from . import autonas, config, hparams, mode, plugins, utils
from .algorithms import *
from .conversion import *
from .utils import *
