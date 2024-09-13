# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Module implementing ``mcore_gpt_minitron`` pruning algorithm for NVIDIA Megatron-Core / NeMo models.

Minitron pruning algorithm uses activation magnitudes to estimate importance of neurons / attention heads in the model.
More details on Minitron pruning algorithm can be found here: https://arxiv.org/pdf/2407.14679
"""
from typing import Optional

import torch

from modelopt.torch.nas.utils import sort_parameters
from modelopt.torch.opt.searcher import BaseSearcher, SearchConfig, SearchStateDict
from modelopt.torch.opt.utils import named_hparams

try:
    from megatron.core.transformer.module import MegatronModule

    HAS_MCORE = True
except Exception:
    HAS_MCORE = False

try:
    from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel

    HAS_NEMO = True
except Exception:
    HAS_NEMO = False


class MCoreGPTMinitronSearcher(BaseSearcher):
    """Searcher for Minitron pruning algorithm."""

    SUPPORTED_HPARAMS = {"ffn_hidden_size", "num_attention_heads", "num_query_groups"}

    @property
    def default_search_config(self) -> SearchConfig:
        """Get the default config for the searcher."""
        return {**super().default_search_config, "max_iter_data_loader": 1024}

    @property
    def default_state_dict(self) -> SearchStateDict:
        """Return default state dict."""
        return {}  # Not used

    def sanitize_search_config(self, config: Optional[SearchConfig]) -> SearchConfig:
        """Sanitize the search config dict."""
        config = super().sanitize_search_config(config)
        assert (
            config["data_loader"] or config["forward_loop"]
        ), "Data loader or forward loop must be provided for importance estimation!"
        return config

    def before_search(self) -> None:
        """Optional pre-processing steps before the search."""
        super().before_search()

        # Check that the constraint is valid
        assert self.constraints.keys() == {
            "export_config"
        }, "Only `export_config` constraint is supported for pruning!"

        export_config = self.constraints["export_config"]
        assert isinstance(export_config, dict)  # to keep mypy happy
        assert (
            export_config.keys() <= MCoreGPTMinitronSearcher.SUPPORTED_HPARAMS
        ), f"Only {MCoreGPTMinitronSearcher.SUPPORTED_HPARAMS} are supported for pruning!"

        assert ("num_attention_heads" in export_config and "num_query_groups" in export_config) or (
            "num_attention_heads" not in export_config and "num_query_groups" not in export_config
        ), "Both `num_attention_heads` and `num_query_groups` should be provided together!"

        # Convert `num_attention_heads` to `num_heads_per_group`
        # Still keep `num_attention_heads` for updating model_cfg below
        if "num_attention_heads" in export_config and "num_query_groups" in export_config:
            export_config["num_heads_per_group"] = (
                export_config["num_attention_heads"] // export_config["num_query_groups"]
            )

        for n, hp in named_hparams(self.model, configurable=True):
            hp_name = n.split(".")[-1]
            if hp_name in export_config:
                assert (
                    export_config[hp_name] in hp.choices
                ), f"Invalid choice for {hp_name}! Available choices: {hp.choices}"

    def run_search(self) -> None:
        """Run actual search."""
        # Run forward loop to collect activations and sort parameters
        assert self.forward_loop is not None
        is_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            self.forward_loop(self.model)
        sort_parameters(self.model)
        self.model.train(is_training)

        # Prune homogeneously
        export_config = self.constraints["export_config"]
        assert isinstance(export_config, dict)  # to keep mypy happy
        for n, hp in named_hparams(self.model, configurable=True):
            hp_name = n.split(".")[-1]
            if hp_name in export_config:
                hp.active = export_config[hp_name]

        # Update model configs so they can be used for restoring the model
        if HAS_MCORE and isinstance(self.model, MegatronModule):
            model_cfg = self.model.config
        elif HAS_NEMO and isinstance(self.model, MegatronGPTModel):
            model_cfg = self.model.cfg
        else:
            raise NotImplementedError("Only MegatronCore and NeMo GPT models are supported!")

        # kv_channels can be None so we need to save original from hidden_size and original num_attention_heads
        orig_kv_channels = getattr(model_cfg, "kv_channels")
        if orig_kv_channels is None:
            orig_kv_channels = getattr(model_cfg, "hidden_size") // getattr(
                model_cfg, "num_attention_heads"
            )
        setattr(model_cfg, "kv_channels", orig_kv_channels)
        for n in MCoreGPTMinitronSearcher.SUPPORTED_HPARAMS:
            if n in export_config:
                setattr(model_cfg, n, export_config[n])
