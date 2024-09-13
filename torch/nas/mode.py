# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Module implementing and describing modes that can be used during the NAS convert process.

Check out :meth:`mtn.convert <modelopt.torch.nas.conversion.convert>` to learn more about modes.
"""

from typing import Optional, Set, Type

from modelopt.torch.opt.config import ModeloptBaseConfig
from modelopt.torch.opt.mode import (
    ConvertEntrypoint,
    RestoreEntrypoint,
    UpdateEntrypoint,
    _ModeDescriptor,
    _ModeRegistryCls,
)
from modelopt.torch.opt.searcher import BaseSearcher

from .autonas import (
    EvolveSearcher,
    convert_autonas_searchspace,
    export_searchspace,
    restore_autonas_searchspace,
    restore_export,
    update_autonas_metadata,
)
from .config import AutoNASConfig, ExportConfig

__all__ = ["AutoNASModeDescriptor", "ExportModeDescriptor"]

NASModeRegistry = _ModeRegistryCls()


@NASModeRegistry.register_mode
class AutoNASModeDescriptor(_ModeDescriptor):
    """Class to describe the ``"autonas"`` mode.

    The properties of this mode can be inspected via the source code.
    """

    @property
    def name(self) -> str:
        """Returns the value (str representation) of the mode."""
        return "autonas"

    @property
    def config_class(self) -> Type[ModeloptBaseConfig]:
        """Specifies the config class for the mode."""
        return AutoNASConfig

    @property
    def next_modes(self) -> Optional[Set[str]]:
        """Modes that must immediately follow this mode."""
        return {"export", "kd_loss", "quantize", "sparse_magnitude", "sparse_gpt"}

    @property
    def export_mode(self) -> Optional[str]:
        """The mode that corresponds to the export mode of this mode."""
        return "export"

    @property
    def search_algorithm(self) -> Type[BaseSearcher]:
        """Specifies the search algorithm to use for this mode (if any)."""
        return EvolveSearcher

    @property
    def convert(self) -> ConvertEntrypoint:
        """The mode's entrypoint for converting a model."""
        return convert_autonas_searchspace

    @property
    def restore(self) -> RestoreEntrypoint:
        """The mode's entrypoint for restoring a model."""
        return restore_autonas_searchspace

    @property
    def update_for_save(self) -> UpdateEntrypoint:
        """The mode's entrypoint for updating the models state before saving."""
        return update_autonas_metadata

    @property
    def update_for_new_mode(self) -> UpdateEntrypoint:
        """The mode's entrypoint for updating the models state before new mode."""
        return update_autonas_metadata


@NASModeRegistry.register_mode
class ExportModeDescriptor(_ModeDescriptor):
    """Class to describe the ``"export"`` mode.

    The properties of this mode can be inspected via the source code.
    """

    @property
    def name(self) -> str:
        """Returns the value (str representation) of the mode."""
        return "export"

    @property
    def config_class(self) -> Type[ModeloptBaseConfig]:
        """Specifies the config class for the mode."""
        return ExportConfig

    @property
    def is_export_mode(self) -> bool:
        """Whether the mode is an export mode.

        Returns:
            True if the mode is an export mode, False otherwise. Defaults to False.
        """
        return True

    @property
    def convert(self) -> ConvertEntrypoint:
        """The mode's entrypoint for converting a model."""
        return export_searchspace

    @property
    def restore(self) -> RestoreEntrypoint:
        """The mode's entrypoint for restoring a model."""
        return restore_export
