# -*- coding: utf-8 -*-

from torch.nn import Module
from torch.nn.functional import interpolate

from ..config import Config


class Interpolate(Module):
    """Wrapper of torch.nn.functionals.interpolate

    """
    def __init__(self, size=None, scale_factor=None, mode='nearest',
                 align_corners=None):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

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
        info += ', align_corners=' + str(self.align_corners)
        return info


def create_pool(kernel_size, **kwargs):
    config = Config()
    paras = config.pool.copy()
    paras.pop('name')
    if config.pool['name'] == 'max':
        if config.dim == 2:
            from torch.nn import MaxPool2d
            return MaxPool2d(kernel_size, **paras, **kwargs)
        elif config.dim == 3:
            from torch.nn import MaxPool3d
            return MaxPool3d(kernel_size, **paras, **kwargs)
    elif config.pool['name'] == 'avg':
        if config.dim == 2:
            from torch.nn import AvgPool2d
            return AvgPool2d(kernel_size, **paras, **kwargs)
        elif config.dim == 3:
            from torch.nn import AvgPool3d
            return AvgPool3d(kernel_size, **paras, **kwargs)


def create_three_pool(**kwargs):
    return create_pool(3, **kwargs)


def create_global_pool(**kwargs):
    config = Config()
    if config.global_pool == 'max':
        if config.dim == 2:
            from torch.nn import AdaptiveMaxPool2d
            return AdaptiveMaxPool2d(1, **kwargs)
        elif config.dim == 3:
            from torch.nn import AdaptiveMaxPool3d
            return AdaptiveMaxPool3d(1, **kwargs)
    elif configs.global_pool == 'avg':
        if config.dim == 2:
            from torch.nn import AdaptiveAvgPool2d
            return AdaptiveAvgPool2d(1)
        elif config.dim == 3:
            from torch.nn import AdaptiveAvgPool3d
            return AdaptiveAvgPool3d(1)


def create_interpolate(size=None, scale_factor=None):
    config = Config()
    if config.upsample['name'] == 'linear':
        if config.dim == 2:
            mode = 'bilinear'
        elif config.dim == 3:
            mode = 'trilinear'
    elif config.upsample['name'] == 'nearest':
        mode = 'nearest'
        config.upsample['align_corners'] = None
    return Interpolate(size=size, scale_factor=scale_factor, mode=mode,
                       align_corners=config.upsample.get('align_corners'))


def create_two_upsample():
    return create_interpolate(scale_factor=2)
