#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ..config import Config


def create_conv(in_channels, out_channels, kernel_size, **kwargs):
    config = Config()
    if config.dim == 2:
        from torch.nn import Conv2d
        return Conv2d(in_channels, out_channels, kernel_size, **kwargs)
    elif config.dim == 3:
        from torch.nn import Conv3d
        return Conv3d(in_channels, out_channels, kernel_size, **kwargs)


def create_conv_trans(in_channels, out_channels, kernel_size, **kwargs):
    config = Config()
    if config.dim == 2:
        from torch.nn import ConvTranspose2d
        return ConvTranspose2d(*args, **kwargs)
    elif config.dim == 3:
        from torch.nn import ConvTranspose2d
        return ConvTranspose3d(*args, **kwargs)


def create_proj(in_channels, out_channels, **kwargs):
    return create_conv(in_channels, out_channels, 1, **kwargs)


def create_three_conv(in_channels, out_channels, **kwargs):
    return create_conv(in_channels, out_channels, 3, padding=1, **kwargs)
