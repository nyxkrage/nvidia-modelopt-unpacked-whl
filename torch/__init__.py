# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Model optimization and deployment subpackage for torch."""

try:
    from . import distill, nas, opt, prune, quantization, sparsity, speculative, utils  # noqa: E402
except ImportError as e:
    raise ImportError("Please install optional ``[torch]`` dependencies.") from e
