# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Pytorch plugins needed for NAS and dynamic modules."""

from typing import Type

import torch.nn as nn

from ..modules import _DynamicBatchNorm
from ..registry import DMRegistry

__all__ = []


def wrapped_convert_sync_batchnorm(cls: Type[nn.Module], /, module: nn.Module, *args, **kwargs):
    """Extend the original convert_sync_batchnorm to handle dynamic modules.

    This method ensures that _DynamicBatchNorm instances are correctly
    converted into DynamicSyncBatchNorm instances.

    .. note::

        We explicitly use ``del`` here following the original implementation!
    """
    # handle vanilla case
    if not isinstance(module, _DynamicBatchNorm):
        return cls.old_convert_sync_batchnorm(module, *args, **kwargs)

    # maintain hparam objects
    hparams = dict(module.named_hparams())

    # set all hparams to max value and retain old active values
    val_actives = {name: hp.active for name, hp in hparams.items()}
    for hp in hparams.values():
        hp.active = hp.max

    # export current module
    module = module.export()

    # convert exported DynamicBN to SyncBN using the original method
    module = cls.old_convert_sync_batchnorm(module, *args, **kwargs)

    # ensure that we indeed got a SyncBN
    assert isinstance(module, nn.SyncBatchNorm), f"Expected SyncBatchNorm, got {type(module)}!"

    # convert SyncBN to DynamicSyncBN
    module = DMRegistry.convert(module)

    # re-use hparams from original dynamic BN to maintain consistency and re-assign active val
    for hp_name, hp in hparams.items():
        setattr(module, hp_name, hp)
        hp.active = val_actives[hp_name]

    # return the new module
    return module


# hook into convert_sync_batchnorm
nn.SyncBatchNorm.old_convert_sync_batchnorm = nn.SyncBatchNorm.convert_sync_batchnorm
nn.SyncBatchNorm.convert_sync_batchnorm = classmethod(wrapped_convert_sync_batchnorm)
