# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


"""User-facing API for converting a model into a `modelopt.torch.speculative.MedusaModel`."""


import torch.nn as nn

from modelopt.torch.opt.conversion import apply_mode
from modelopt.torch.opt.mode import ModeLike

from .mode import SpeculativeDecodingModeRegistry

__all__ = ["convert"]


def convert(model: nn.Module, mode: ModeLike) -> nn.Module:
    """Main conversion function to turn a base model into a speculative decoding model.

    Args:
        model: The base model to be used.
        mode: A (list of) string(s) or Mode(s) or a list of tuples containing the mode and its
            config indicating the desired mode(s) (and configurations) for the convert
            process. Modes set up the model for different algorithms for model optimization. The
            following modes are available:

            *   :class:`"medusa"<modelopt.torch.speculative.mode.MedusaModeDescriptor>`: The
                ``model`` will be converted into a medusa model with added medusa head.
                The mode's config is described in
                :class:`MedusaConfig<modelopt.torch.speculative.config.MedusaConfig>`.

            If the mode argument is specified as a dictionary, the keys should indicate the mode and
            the values specify the per-mode configuration.

    Returns:
        An instance of :class:`MedusaModel <modelopt.torch.distill.MedusaModel` or its subclass.

    """
    return apply_mode(model, mode=mode, registry=SpeculativeDecodingModeRegistry)
