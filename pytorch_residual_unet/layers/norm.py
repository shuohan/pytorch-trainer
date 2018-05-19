# -*- coding: utf-8 -*-

from torch.nn import InstanceNorm2d, InstanceNorm3d, BatchNorm2d, BatchNorm3d

from ..config import Configuration


config = Configuration()

if config.num_dims == 2:
    InstanceNorm_ = InstanceNorm2d
    BatchNorm_ = BatchNorm2d
elif config.num_dims == 3:
    InstanceNorm_ = InstanceNorm3d
    BatchNorm_ = BatchNorm3d


class InstanceNorm(InstanceNorm_):
    """Customized instance normalization"""
    def __init__(self, num_features):
        super().__init__(num_features, affine=True, track_running_stats=True)


class BatchNorm(BatchNorm_):
    """Customized batch normalization"""
    def __init__(self, num_features):
        super().__init__(num_features, affine=True, track_running_stats=True)


if config.norm == 'instance':
    Normalization = InstanceNorm
elif config.norm == 'batch':
    Normalization = BatchNorm
