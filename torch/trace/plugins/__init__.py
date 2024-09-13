# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Handles tracing plugins for third-party modules."""
import warnings as _warnings

try:
    from .megatron import *

except ImportError:
    pass
except Exception as e:
    _warnings.warn(f"Failed to import megatron plugin due to: {repr(e)}")

try:
    from .transformer_engine import *

except ImportError:
    pass
except Exception as e:
    _warnings.warn(f"Failed to import transformer engine plugin due to: {repr(e)}")

try:
    from .transformers import *

except ImportError:
    pass
except Exception as e:
    _warnings.warn(f"Failed to import transformers plugin due to: {repr(e)}")
