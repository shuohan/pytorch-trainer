# -*- coding: utf-8 -*-

from torch.nn import Module
from torch.nn import MaxPool2d, MaxPool3d, AvgPool2d, AvgPool3d
from torch.nn import AdaptiveMaxPool2d, AdaptiveMaxPool3d
from torch.nn import AdaptiveAvgPool2d, AdaptiveAvgPool3d
from torch.nn.functional import interpolate

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


class Upsample(Module):
    """Customized Upsample

    See torch.nn.Upsample for more details.

    """
    def __init__(self, size=None, scale_factor=None):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        if config.upsample_mode == 'linear':
            if config.num_dims == 2:
                self.mode = 'bilinear'
            elif config.num_dims == 3:
                self.mode = 'trilinear'
        elif config.upsample_mode == 'nearest':
            self.mode = 'nearest'
        self.align_corners = config.upsample_align_corners
        if 'linear' not in self.mode:
            self.align_corners = None

    def forward(self, input):
        output = interpolate(input, size=self.size,
                             scale_factor=self.scale_factor,
                             mode=self.mode, align_corners=self.align_corners)
        return output

    def extra_repr(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info
