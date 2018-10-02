# -*- coding: utf-8 -*-

from torch.nn import MaxPool2d, MaxPool3d, AvgPool2d, AvgPool3d
from torch.nn import AdaptiveMaxPool2d, AdaptiveMaxPool3d
from torch.nn import AdaptiveAvgPool2d, AdaptiveAvgPool3d
from torch.nn import Upsample as Upsample_

from ..config import Configuration


config = Configuration()

if config.num_dims == 2:
    MaxPool = MaxPool2d
    AvgPool = AvgPool2d
    AdaptiveMaxPool = AdaptiveMaxPool2d
    AdaptiveAvgPool = AdaptiveAvgPool2d
elif config.num_dims == 3:
    MaxPool = MaxPool3d
    AvgPool = AvgPool3d
    AdaptiveMaxPool = AdaptiveMaxPool3d
    AdaptiveAvgPool = AdaptiveAvgPool3d

if config.pool == 'max':
    Pool = MaxPool
elif config.pool == 'average':
    Pool = AvgPool

if config.global_pool == 'max':
    AdaptivePool = AdaptiveMaxPool
elif config.global_pool == 'average':
    AdaptivePool = AdaptiveAvgPool

class GlobalPool(AdaptivePool):
    """Spatial global pooling"""
    def __init__(self):
        super().__init__(1)


class Upsample(Upsample_):
    """Customized Upsample"""
    def __init__(self, **kwargs) :
        if config.upsample_mode == 'linear':
            if config.num_dims == 2:
                kwargs['mode'] = 'bilinear'
            elif config.num_dims == 3:
                kwargs['mode'] = 'trilinear'
        elif config.upsample_mode == 'nearest':
            kwargs['mode'] = 'nearest'
        super().__init__(**kwargs)
