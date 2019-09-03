# -*- coding: utf-8 -*-
"""Functions to create convolutional layers.

"""
from ..config import Config


def create_conv(in_channels, out_channels, kernel_size, **kwargs):
    """Creates a convolutional layer.

    Note:
        This function supports creating a 2D or 3D convolutional layer
        configured by :attr:`pytorch_engine.Config.dim`.

    Note:
        The function passes all keyword arguments directly to the Conv class.
        Check pytorch documentation for all keyword arguments (``bias``, for
        example).

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): The size of kernel.

    Returns:
        torch.nn.Module: The created convolutional layer.

    """
    if Config.dim == 2:
        from torch.nn import Conv2d
        return Conv2d(in_channels, out_channels, kernel_size, **kwargs)
    elif Config.dim == 3:
        from torch.nn import Conv3d
        return Conv3d(in_channels, out_channels, kernel_size, **kwargs)


def create_conv_trans(in_channels, out_channels, kernel_size, **kwargs):
    """Creates a transposed convolutional layer.

    Check :func:`create_conv` for more details.

    """
    if Config.dim == 2:
        from torch.nn import ConvTranspose2d
        return ConvTranspose2d(*args, **kwargs)
    elif Config.dim == 3:
        from torch.nn import ConvTranspose2d
        return ConvTranspose3d(*args, **kwargs)


def create_proj(in_channels, out_channels, **kwargs):
    """Creates a projection convolutional layer (kernel 1).

    Check :func:`create_conv` for more details.

    """
    return create_conv(in_channels, out_channels, 1, **kwargs)


def create_three_conv(in_channels, out_channels, **kwargs):
    """Creates a convolutional layer with kernel 3 and ``"same"`` padding.

    Check :func:`create_conv` for more details.

    """
    return create_conv(in_channels, out_channels, 3, padding=1, **kwargs)
