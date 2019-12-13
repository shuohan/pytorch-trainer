# -*- coding: utf-8 -*-

from torch.nn import Module
from torch.nn.functional import interpolate

from ..config import Config


class Interpolate(Module):
    """Wrapper of :func:`torch.nn.functionals.interpolate`.

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
    """Creates an pooling layer.

    Note:
        The parameters are configured in :attr:`pytorch_engine.Config.pool`.
        Theres parameters should be mutually exclusive from ``kwargs``.

    Args:
        kernel_size (int or iterable[int]) : The size of the pooling window.
        kwargs (dict): The addtional parameters of pooling. See pytorch MaxPool
            and AvgPool for more details.

    Returns:
        torch.nn.Module: The created pooling layer.

    """
    paras = Config.pool.copy()
    paras.pop('name')
    if Config.pool['name'] == 'max':
        if Config.dim == 2:
            from torch.nn import MaxPool2d
            return MaxPool2d(kernel_size, **paras, **kwargs)
        elif Config.dim == 3:
            from torch.nn import MaxPool3d
            return MaxPool3d(kernel_size, **paras, **kwargs)
    elif Config.pool['name'] == 'avg':
        if Config.dim == 2:
            from torch.nn import AvgPool2d
            return AvgPool2d(kernel_size, **paras, **kwargs)
        elif Config.dim == 3:
            from torch.nn import AvgPool3d
            return AvgPool3d(kernel_size, **paras, **kwargs)


def create_three_pool(**kwargs):
    """Creates pooling with kernel size 3."""
    return create_pool(3, **kwargs)


def create_global_pool(**kwargs):
    """Creates global pooling with the kernel size equal to the image size.

    Note:
        The type of global pooling is configured in
        :attr:`pytorch_engine.Config.global_pool`.

    Returns:
        torch.nn.Module: The created pooling layer.

    """
    if Config.global_pool == 'max':
        if Config.dim == 2:
            from torch.nn import AdaptiveMaxPool2d
            return AdaptiveMaxPool2d(1, **kwargs)
        elif Config.dim == 3:
            from torch.nn import AdaptiveMaxPool3d
            return AdaptiveMaxPool3d(1, **kwargs)
    elif Config.global_pool == 'avg':
        if Config.dim == 2:
            from torch.nn import AdaptiveAvgPool2d
            return AdaptiveAvgPool2d(1)
        elif Config.dim == 3:
            from torch.nn import AdaptiveAvgPool3d
            return AdaptiveAvgPool3d(1)


def create_interpolate(size=None, scale_factor=None):
    """Creates a interpolate layer.

    See :func:`torch.nn.functionals.interpolate` for the inputs ``size`` and
    ``scale_factor``.

    Note:
        The type and other parameters of interpolate are configured in
        :attr:`pytorch_engine.Config.upsample`.

    Returns:
        torch.nn.Module: The created interpolate layer.

    """
    if Config.upsample['name'] == 'linear':
        if Config.dim == 2:
            mode = 'bilinear'
        elif Config.dim == 3:
            mode = 'trilinear'
    elif Config.upsample['name'] == 'nearest':
        mode = 'nearest'
        Config.upsample['align_corners'] = None
    return Interpolate(size=size, scale_factor=scale_factor, mode=mode,
                       align_corners=Config.upsample.get('align_corners'))


def create_two_upsample():
    """Creates interpolate with scale factor 2."""
    return create_interpolate(scale_factor=2)
