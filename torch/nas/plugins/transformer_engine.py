# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Plugin to add NAS support for Transformer Engine modules."""

import transformer_engine as te

from ..modules import _DynamicLayerNorm
from ..registry import DMRegistry
from ..traced_hp import TracedHp

__all__ = ["_DynamicTENorm"]


@DMRegistry.register(
    {te.pytorch.LayerNorm: "te.pytorch.LayerNorm", te.pytorch.RMSNorm: "te.pytorch.RMSNorm"}
)
class _DynamicTENorm(_DynamicLayerNorm):
    """A ``te.pytorch.{Layer/RMS}Norm`` layer with dynamic hyperparams."""

    def _setup(self):
        hidden_size = self.weight.shape[-1]

        # register the hyperparameter with a new name
        self._register_hparam("num_features", TracedHp(list(range(1, hidden_size + 1))))

        # register dynamic attributes
        self._register_dynamic_attribute("weight", self._cut_to_active_features)
        if hasattr(self, "bias"):  # Bias is not present in RMSNorm
            self._register_dynamic_attribute("bias", self._cut_to_active_features)
