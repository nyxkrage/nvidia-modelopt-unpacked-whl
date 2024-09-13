# Adapted from: https://github.com/ctlllll/axolotl/blob/f86767e/src/axolotl/monkeypatch/medusa_utils.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Support medusa for huggingface models."""

import contextlib
import math
import os
import types
import warnings
from typing import Any, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel
from transformers.modeling_utils import load_sharded_checkpoint, load_state_dict
from transformers.trainer_pt_utils import LabelSmoother

from ..medusa.conversion import MedusaDMRegistry
from ..medusa.medusa_model import MedusaModel, ResBlock

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@MedusaDMRegistry.register({PreTrainedModel: "hf.PreTrainedModel"})
class HFMedusaModel(MedusaModel):
    """Medusa Model Class for huggingface models."""

    def modify(self, medusa_num_heads=0, medusa_num_layers=0):
        """Constructor.

        Args:
            config: MedusaConfig that specifies the medusa head configuration as well as
                    weights of base model and medusa head.
        """
        super().modify(medusa_num_heads=medusa_num_heads, medusa_num_layers=medusa_num_layers)

        hidden_size = self.lm_head.weight.shape[-1]
        vocab_size = self.lm_head.weight.shape[0]

        # Create a list of Medusa heads
        self.medusa_head = nn.ModuleList(
            [
                nn.Sequential(
                    *([ResBlock(hidden_size)] * self.medusa_num_layers),
                    nn.Linear(hidden_size, vocab_size, bias=False),
                )
                for _ in range(self.medusa_num_heads)
            ]
        )

        # Ensure medusa_head's dtype and device align with the base_model
        self.medusa_head.to(self.lm_head.weight.dtype).to(self.lm_head.weight.device)
        self.medusa_head.device = self.lm_head.weight.device
        if hasattr(self, "hf_device_map") and "lm_head" in self.hf_device_map:
            self.hf_device_map["medusa_head"] = self.hf_device_map["lm_head"]

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        medusa_return: bool = True,
        medusa_only_heads: bool = True,
    ) -> Any:
        """Forward pass of the MedusaModel.

        Returns:
            torch.Tensor: A tensor containing predictions from all Medusa heads.
        """
        if not medusa_return:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # Pass input through the base model
        with torch.no_grad() if medusa_only_heads else contextlib.nullcontext():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = outputs[0]
            medusa_logits = [self.lm_head(hidden_states)]
        for i in range(self.medusa_num_heads):
            medusa_logits.append(self.medusa_head[i](hidden_states))
        return torch.stack(medusa_logits, dim=0)


def replace_medusa_compute_loss(
    self,
    medusa_heads_coefficient=0.2,
    medusa_decay_coefficient=0.8,
    medusa_scheduler="constant",
    medusa_only_heads=True,
    medusa_distillation_regularization=0.0,
    medusa_self_distillation=False,
):
    """Replace compute_loss in HF trainer for Medusa.

    Args:
    medusa_heads_coefficient (float): The coefficient for overall medusa loss.
    medusa_decay_coefficient (float): The decay coefficient for each medusa head loss.
    medusa_scheduler (str): The type of scheduler for medusa loss decay.
    medusa_only_heads (bool): Whether only fine-tune medusa heads and freeze base model.
    medusa_distillation_regularization (float): medusa self distillation regularization coefficient.
    medusa_self_distillation (bool): Whether to use medusa self distillation.
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute the training loss for the model.

        Args:
            model (torch.nn.Module): The model for which to compute the loss.
            inputs (dict): The input data, including input IDs, attention mask, and labels.
            return_outputs (bool): Whether to return model outputs along with the loss.

        Returns:
            Union[float, Tuple[float, torch.Tensor]]: The computed loss, optionally with model outputs.
        """
        if self.label_smoother is not None:
            warnings.warn(
                "label_smoother is enabled. However, label smoothing is not supported with medusa loss..."
            )
        if medusa_self_distillation:
            from peft.tuners.tuners_utils import BaseTunerLayer

            with torch.inference_mode():
                # Get the output of the original model for distillation
                for module in model.modules():
                    if isinstance(module, (BaseTunerLayer)):
                        module.enable_adapters(False)

                original_logits = model(
                    **inputs,
                    medusa_return=False,
                ).logits

                for module in model.modules():
                    if isinstance(module, (BaseTunerLayer)):
                        module.enable_adapters(True)

        logits = model(
            **inputs,
            medusa_return=True,
            medusa_only_heads=medusa_only_heads,
        )
        labels = inputs["labels"]
        # Shift so that tokens < n predict n
        loss = 0
        loss_fct = CrossEntropyLoss()
        log = {}
        medusa = logits.shape[0]
        for i in range(medusa):
            medusa_logits = logits[i, :, : -(1 + i)].contiguous()
            medusa_labels = labels[..., 1 + i :].contiguous()
            medusa_logits = medusa_logits.view(-1, logits.shape[-1])
            medusa_labels = medusa_labels.view(-1)
            medusa_labels = medusa_labels.to(medusa_logits.device)
            if i == 0:
                if medusa_self_distillation:
                    original_logits = (
                        original_logits[:, :-1].contiguous().view(-1, original_logits.shape[-1])
                    )
                    mask = medusa_labels.ne(IGNORE_TOKEN_ID)
                    soft_labels = F.softmax(original_logits[mask], dim=-1)
                    loss_i = (
                        F.kl_div(
                            F.log_softmax(medusa_logits[mask], dim=-1),
                            soft_labels,
                            reduction="sum",
                        )
                        / medusa_logits.shape[0]
                    )
                elif medusa_distillation_regularization > 0:
                    # use soft labels
                    mask = medusa_labels.ne(IGNORE_TOKEN_ID)
                    soft_labels = F.softmax(
                        medusa_logits[mask], dim=-1
                    ) * medusa_distillation_regularization + F.one_hot(
                        medusa_labels[mask], num_classes=medusa_logits.shape[-1]
                    ) * (
                        1 - medusa_distillation_regularization
                    )
                    loss_i = (
                        F.kl_div(
                            F.log_softmax(medusa_logits[mask], dim=-1),
                            soft_labels,
                            reduction="sum",
                        )
                        / medusa_logits.shape[0]
                    )
                else:
                    loss_i = loss_fct(medusa_logits, medusa_labels)
            else:
                loss_i = loss_fct(medusa_logits, medusa_labels)
            # Compute the coefficient for medusa losses
            if medusa_scheduler == "sine":
                medusa_scheduler_coefficient = math.sin(
                    self.state.global_step / self.state.max_steps * math.pi / 2
                )
            elif medusa_scheduler == "linear":
                medusa_scheduler_coefficient = self.state.global_step / self.state.max_steps
            elif medusa_scheduler == "constant":
                medusa_scheduler_coefficient = 1
            elif medusa_scheduler.startswith("sine"):
                ratio = float(medusa_scheduler.split("_")[1])
                if self.state.global_step / self.state.max_steps < ratio:
                    medusa_scheduler_coefficient = math.sin(
                        self.state.global_step / self.state.max_steps / ratio * math.pi / 2
                    )
                else:
                    medusa_scheduler_coefficient = 1
            else:
                raise ValueError(
                    f"Invalid medusa_scheduler: {medusa_scheduler}. "
                    "Must be one of 'sine', 'linear', or 'constant'."
                )
            # Add decay coefficient to the loss
            if i == 0:
                if not medusa_only_heads:
                    loss += loss_i
            else:
                loss += (
                    loss_i
                    * medusa_decay_coefficient**i
                    * medusa_heads_coefficient
                    * medusa_scheduler_coefficient
                )
            not_ignore = medusa_labels.ne(IGNORE_TOKEN_ID)
            medusa_labels = medusa_labels[not_ignore]

            # Add top-k accuracy
            for k in range(1, 10):
                _, topk = medusa_logits.topk(k, dim=-1)
                topk = topk[not_ignore]
                correct = topk.eq(medusa_labels.unsqueeze(-1)).any(-1)
                log[f"medusa{i}_top{k}"] = correct.float().mean().item()

            log[f"medusa{i}_loss"] = loss_i.item()
            log["medusa_scheduler_coefficient"] = medusa_scheduler_coefficient
        # self.log(log)
        # Add prefix to the log
        if model.training:
            prefix = "train"
        else:
            prefix = "eval"
        log = {f"{prefix}/{k}": v for k, v in log.items()}
        return (loss, logits) if return_outputs else loss

    self.compute_loss = types.MethodType(compute_loss, self)


def load_medusa_head(model, medusa_head_path):
    """Load medusa head weight."""
    print("Loading medusa head from ", medusa_head_path)
    if os.path.isfile(medusa_head_path):
        medusa_head = torch.load(medusa_head_path, map_location="cpu")
        model.medusa_head.load_state_dict(medusa_head)
    elif os.path.isdir(medusa_head_path):
        try:
            load_sharded_checkpoint(model, medusa_head_path)
        except Exception as e:
            print(e)
            if os.path.isfile(os.path.join(medusa_head_path, "pytorch_model.bin")):
                state_dict = load_state_dict(os.path.join(medusa_head_path, "pytorch_model.bin"))
                model.load_state_dict(state_dict)
                print(
                    "Medusa head loaded from ",
                    os.path.join(medusa_head_path, "pytorch_model.bin"),
                )
            elif os.path.isfile(os.path.join(medusa_head_path, "model.safetensors")):
                state_dict = load_state_dict(os.path.join(medusa_head_path, "model.safetensors"))
                model.load_state_dict(state_dict)
                print(
                    "Medusa head loaded from ",
                    os.path.join(medusa_head_path, "model.safetensors"),
                )
