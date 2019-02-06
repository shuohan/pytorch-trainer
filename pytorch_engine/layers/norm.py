# -*- coding: utf-8 -*-

from torch.nn import InstanceNorm2d, InstanceNorm3d, BatchNorm2d, BatchNorm3d

from ..configs import Configurations


configs = Configurations()

if configs.num_dims == 2:
    InstanceNorm = InstanceNorm2d
    BatchNorm = BatchNorm2d
elif configs.num_dims == 3:
    InstanceNorm = InstanceNorm3d
    BatchNorm = BatchNorm3d

if configs.norm == 'instance':
    Normalization = InstanceNorm
elif configs.norm == 'batch':
    Normalization = BatchNorm
