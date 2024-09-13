# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""ModelOpt plugin for enabling automatic save/restore of ModelOpt state for HuggingFace models."""

import functools
import os
import types
import warnings

import torch

from modelopt.torch.utils import print_rank_0

from ..conversion import ModeloptStateManager, modelopt_state, restore_from_modelopt_state

__all__ = ["enable_huggingface_checkpointing"]


def _get_modelopt_state_path(model_name_or_path):
    return os.path.join(model_name_or_path, "modelopt_state.pth")


def _new_from_pretrained(cls, /, pretrained_model_name_or_path, *args, **kwargs):

    cls._original__init__ = cls.__init__

    @functools.wraps(cls._original__init__)
    def new_init_fn(self, *args, **kwargs):
        modelopt_state_path = _get_modelopt_state_path(pretrained_model_name_or_path)
        cls._original__init__(self, *args, **kwargs)
        if os.path.isfile(modelopt_state_path):
            restore_from_modelopt_state(self, torch.load(modelopt_state_path, map_location="cpu"))
            print_rank_0(f"Restored ModelOpt state from {modelopt_state_path}")

    cls.__init__ = new_init_fn
    model = types.MethodType(cls._modelopt_cache["from_pretrained"].__func__, cls)(
        pretrained_model_name_or_path, *args, **kwargs
    )
    cls.__init__ = cls._original__init__
    delattr(cls, "_original__init__")
    return model


def _new_save_pretrained(self, save_directory, *args, **kwargs):
    self._modelopt_cache["save_pretrained"](self, save_directory, *args, **kwargs)
    if ModeloptStateManager.is_converted(self):
        torch.save(modelopt_state(self), _get_modelopt_state_path(save_directory))
        print_rank_0(f"Saved ModelOpt state to {_get_modelopt_state_path(save_directory)}")


def _patch_pretrained_methods(cls):
    if hasattr(cls, "_modelopt_cache"):
        return
    cls._modelopt_cache = {
        "from_pretrained": cls.from_pretrained,
        "save_pretrained": cls.save_pretrained,
    }
    cls.from_pretrained = classmethod(_new_from_pretrained)
    cls.save_pretrained = _new_save_pretrained


try:
    from transformers import PreTrainedModel

    _patch_pretrained_methods(PreTrainedModel)
except ImportError:
    pass

try:
    from diffusers import ModelMixin

    _patch_pretrained_methods(ModelMixin)
except ImportError:
    pass


def enable_huggingface_checkpointing():
    """Enables automatic save/restore of ModelOpt state with HuggingFace checkpointing APIs.

    ModelOpt automatically saves `modelopt_state` to `save_directory/modelopt_state.pth` when
    a Huggingface model is saved using
    `model.save_pretrained(save_directory) <https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.save_pretrained>`_.

    Conversely, ModelOpt restores the saved state from `pretrained_model_name_or_path/modelopt_state.pth` if it exists
    when a Huggingface model is loaded using
    `cls.from_pretrained(pretrained_model_name_or_path) <https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained>`_.


    This function should be called once in the program before loading/saving any HuggingFace models.

    Here is an example usage:

    .. code-block:: python

        from transformers import AutoModelForCausalLM
        import modelopt.torch.opt as mto

        # Enable ModelOpt save/restore for HuggingFace models
        # This only needs to be called once in the program.
        mto.enable_huggingface_checkpointing()

        # Instantiate a HuggingFace model, modelopt_state will be automatically loaded if it exists.
        model = AutoModelForCausalLM.from_pretrained(model_path).cuda()

    """
    # This method simply prints if ModelOpt save/restore is enabled for the HuggingFace libraries.
    try:
        from transformers import PreTrainedModel

        assert hasattr(PreTrainedModel, "_modelopt_cache")
        print_rank_0("ModelOpt save/restore enabled for `transformers` library.")
    except ImportError:
        pass

    try:
        from diffusers import ModelMixin

        assert hasattr(ModelMixin, "_modelopt_cache")
        print_rank_0("ModelOpt save/restore enabled for `diffusers` library.")
    except ImportError:
        pass

    # TODO: Add support for PEFT library
    try:
        import peft  # noqa

        warnings.warn("ModelOpt does not support automatic save/restore for `peft` library.")
    except ImportError:
        pass
