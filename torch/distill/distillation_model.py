# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Meta-model wrapper to support knowledge-distillation learning."""

import inspect
import warnings
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss as Loss

from modelopt.torch.opt.dynamic import DynamicModule

from .loss_balancers import STUDENT_LOSS_KEY, DistillationLossBalancer

__all__ = ["DistillationModel"]

# Avoid multiple printouts of warnings in train loop.
warnings.simplefilter("once")


class DistillationModel(DynamicModule):
    """Class to encapsulate multiple teacher and student models as a single model."""

    def _setup(self):
        self._register_temp_attribute("_layers_to_loss", {})
        self._register_temp_attribute("_loss_balancer", None)
        self._register_temp_attribute("_expose_minimal_state_dict", None)
        self._register_temp_attribute("_teacher_model", None)
        self._register_temp_attribute("_loss_modules", nn.ModuleList())

        # HACK: set model's forward signature to match student class' original.
        # Needed for HF `transformers.utils.find_labels` which relies on inspecting class signature.
        sig_old = inspect.signature(self.original_cls.forward)
        sig_new = inspect.signature(type(self).forward)
        type(self).forward.__signature__ = sig_new.replace(  # type: ignore[attr-defined]
            parameters=tuple(sig_old.parameters.values()),
            return_annotation=sig_old.return_annotation,
        )

    def modify(
        self,
        teacher_model: nn.Module,  # To be frozen.
        criterion: Dict[
            Tuple[
                str,  # Student model layer whose output to capture.
                str,  # Teacher model layer whose output to capture.
            ],
            Loss,  # Loss fn.
        ],
        loss_balancer: Optional[DistillationLossBalancer] = None,
        expose_minimal_state_dict: bool = True,
    ):
        """Constructor.

        Args:
            teacher_model: A teacher model which this class would encapsulate.
            criterion: A dictionary mapping the tuple of student and teacher
                model layer names to the loss function to apply to that layer pair.
            loss_balancer: Instance of
                :class:`DistillationLossBalancer <modelopt.torch.distill.DistillationLossBalancer>`
                which reduces distillation and non-distillation losses into a single value using some weighing scheme.
            expose_minimal_state_dict: If True, will hide teacher's state dict when calling ``state_dict`` on this
                class. This allows avoiding to save the teacher state unnecessarily during checkpointing.
                .. note: Set to False if using `FSDP <https://pytorch.org/docs/stable/fsdp.html>`_
        """
        self._loss_balancer = loss_balancer
        self._expose_minimal_state_dict = expose_minimal_state_dict

        # Assign loss to specified modules.
        self._layers_to_loss = {
            (
                self.get_submodule(student_layer_name),
                teacher_model.get_submodule(teacher_layer_name),
            ): loss_fn
            for (
                student_layer_name,
                teacher_layer_name,
            ), loss_fn in criterion.items()
        }

        # Register all child modules not automatically registered with assignment operator.
        # This is done to ensure that the parameters of all the underlying nn.Modules
        # in the DistillationModel appear to the caller when querying for parameters or
        # child modules of DistillationModel object. These might be needed when attaching
        # model params to optimizer, saving/restoring state of DistillationModel etc.
        self._teacher_model = teacher_model
        self._loss_modules = nn.ModuleList(
            set(m for m in self._layers_to_loss.values() if len(list(m.parameters())) > 0)
        )

        # Disable grad for teacher
        self._teacher_model.requires_grad_(False)

        # Register hooks for intermediate outputs from teacher models and the student model.
        # HACK: For inexplicable reasons, sometimes a model will have hooks remain after
        #   `ato.restore()` so we check if they are present accidentally first.
        for student_layer, teacher_layer in self._layers_to_loss:
            if output_capture_fwd_hook not in student_layer._forward_hooks.values():
                student_layer.register_forward_hook(output_capture_fwd_hook)
            if output_capture_fwd_hook not in teacher_layer._forward_hooks.values():
                teacher_layer.register_forward_hook(output_capture_fwd_hook)

    @property
    def teacher_model(self) -> nn.ModuleList:
        """Fetch the teacher model."""
        return self._teacher_model

    @property
    def loss_modules(self) -> nn.ModuleList:
        """Fetch the loss modules list."""
        return self._loss_modules

    @property
    def loss_balancer(self) -> Optional[DistillationLossBalancer]:
        """Fetch the loss balancer, if any."""
        return self._loss_balancer

    @contextmanager
    def hide_teacher_model(self, enable=True):
        """Context manager to temporarily hide teacher model from the model."""
        teacher_model = self._teacher_model
        if enable:
            self._teacher_model = nn.Module()
        try:
            yield
        finally:
            self._teacher_model = teacher_model

    @contextmanager
    def hide_loss_modules(self, enable=True):
        """Context manager to temporarily hide teacher model from the model."""
        loss_modules = self._loss_modules
        if enable:
            self._loss_modules = nn.ModuleList()
        try:
            yield
        finally:
            self._loss_modules = loss_modules

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        """Override to potentially return the state without teacher's."""
        with self.hide_teacher_model(enable=self._expose_minimal_state_dict):
            return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs) -> Any:
        """Override to potentially load the state without teacher's or loss modules'."""
        # Don't expose teacher/loss modules when loading at first from a non-DistillationModel checkpoint.
        hide_teacher = self._expose_minimal_state_dict or not any(
            k.startswith("_teacher_model") for k in state_dict
        )
        hide_losses = len(self._loss_modules) > 0 and not any(
            k.startswith("_loss_modules") for k in state_dict
        )
        with self.hide_teacher_model(enable=hide_teacher):
            with self.hide_loss_modules(enable=hide_losses):
                return super().load_state_dict(state_dict, *args, **kwargs)

    def forward(self, *args, **kwargs) -> Any:
        """Implement forward pass.

        Args:
            *args: Positional inputs to the student and teacher model.
            **kwargs: Named inputs to the student and teacher model.

        Returns:
            The student model's output.
        """
        # Call teacher model's forward pass for layer outputs to get computed.
        # no_grad() context lets pytorch know not to save activations for
        # teacher models in memory as there won't be any gradient updates applied
        # to these layers. This consumes less memory than just freezing teacher model weights.
        with torch.no_grad():
            # Calling `.train()` on this class inadvertently calls it on teacher too.
            self._teacher_model.eval()
            self._teacher_model(*args, **kwargs)

        student_output = super().forward(*args, **kwargs)

        return student_output

    def compute_kd_loss(
        self,
        student_loss: Optional[torch.Tensor] = None,
        loss_reduction_fn: Callable = None,
        skip_balancer: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute total loss for distillation backpropagation.

        Args:
            student_loss: Original loss computed from the student's output.
            loss_reduction_fn: Callable to be called on each loss tensor prior to balancing. Useful for
                loss-masking situations where the callable changes arguments each iteration.
            skip_balancer: Whether or not to use loss balancer to reduce the loss dict into a scalar.

        Returns:
            If reduce is True, the scalar total loss weighted between ``student_loss`` and the distillation losses.
            If reduce is False, a dict of student model output loss and layer-wise distillation losses.
        """
        if self._loss_balancer is None:
            assert student_loss is None, "Cannot pass in student loss without using Loss Balancer."
        loss_dict: Dict[str, torch.Tensor] = {}
        if student_loss is not None:
            loss_dict[STUDENT_LOSS_KEY] = student_loss

        idx = 0
        for (student_layer, teacher_layer), loss_fn in self._layers_to_loss.items():
            out_s = getattr(student_layer, "_intermediate_output")
            out_t = getattr(teacher_layer, "_intermediate_output")
            delattr(student_layer, "_intermediate_output")
            delattr(teacher_layer, "_intermediate_output")

            loss = loss_fn(out_s, out_t)  # Student is pred, Teacher is target
            if loss_reduction_fn is not None:
                # Needed in cases where a loss mask is used on non-scalar loss-fn outputs, prior to
                # reducing to a scalar loss value.
                loss = loss_reduction_fn(loss)
            loss_dict[f"{loss_fn.__class__.__name__}_{idx}"] = loss
            idx += 1

        if skip_balancer:
            # Needed for special case if reduction needs to be done separately before balancing.
            return loss_dict

        if self._loss_balancer is None:
            assert len(loss_dict) == 1  # we ensure this in constructor
            loss_total = next(iter(loss_dict.values()))
        else:
            loss_total = self._loss_balancer(loss_dict)

        return loss_total


def output_capture_fwd_hook(
    module: nn.Module, input: Any, output: Any
):  # pylint: disable=redefined-builtin  # noqa
    """A hook to capture layer output."""
    # NOTE: Defined externally to allow pickling.
    if module.training and hasattr(module, "_intermediate_output"):
        warnings.warn(
            f"Module `{type(module).__name__}` already has an intermediate output stored."
            " This is undesired behavior unless Gradient Checkpointing is in use."
        )
    setattr(module, "_intermediate_output", output)
