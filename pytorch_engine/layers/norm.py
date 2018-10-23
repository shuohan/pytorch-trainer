# -*- coding: utf-8 -*-

from torch.nn import InstanceNorm2d, InstanceNorm3d, BatchNorm2d, BatchNorm3d

from ..config import Configuration


config = Configuration()

if config.num_dims == 2:
    InstanceNorm = InstanceNorm2d
    BatchNorm = BatchNorm2d
elif config.num_dims == 3:
    InstanceNorm = InstanceNorm3d
    BatchNorm = BatchNorm3d

if config.norm == 'instance':
    Normalization = InstanceNorm
elif config.norm == 'batch':
    Normalization = BatchNorm
