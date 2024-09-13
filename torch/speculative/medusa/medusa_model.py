# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Medusa model to support medusa decoding."""


from torch import nn

from modelopt.torch.opt.dynamic import DynamicModule


class ResBlock(nn.Module):
    """A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        """Init function of ResBlock.

        Args:
        hidden_size (int): The size of the hidden layers in the block.
        """
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))


class MedusaModel(DynamicModule):
    """Base Medusa Model."""

    def _setup(self):
        self._register_temp_attribute("medusa_num_heads", 0)
        self._register_temp_attribute("medusa_num_layers", 0)
        self._register_temp_attribute("medusa_head", nn.ModuleList())

    def modify(self, medusa_num_heads=0, medusa_num_layers=0):
        """Base Medusa Model modify function. Child class should implement the details."""
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
