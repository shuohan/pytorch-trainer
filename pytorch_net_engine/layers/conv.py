#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.nn import Conv2d, Conv3d, ConvTranspose2d, ConvTranspose3d

from ..config import Configuration


config = Configuration()
if config.num_dims == 2:
    Conv_ = Conv2d
    ConvTrans_ = ConvTranspose2d
elif config.num_dims == 3:
    Conv_ = Conv3d
    ConvTrans_ = ConvTranspose3d


class Conv(Conv_):
    """Customized convolution layer"""

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        """Initliaze
        
        Do not use bias since convolution is followed by batch/instance
        normalization

        Args:
            in_channels (int): The number of input channels
            out_channels (int): The number of output channels
            kernel_size (int or tuple): The size of convolution kernel
            kwargs (dict): See other settings in PyTorch doc

        """
        super().__init__(in_channels, out_channels, kernel_size, bias=False,
                         **kwargs)

class ConvTranspose(ConvTrans_):
    """Customized transposed convolution"""

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        """Initliaze
        
        Do not use bias since convolution is followed by batch/instance
        normalization

        Args:
            in_channels (int): The number of input channels
            out_channels (int): The number of output channels
            kernel_size (int or tuple): The size of convolution kernel
            kwargs (dict): See other settings in PyTorch doc

        """
        super().__init__(in_channels, out_channels, kernel_size, bias=False,
                         **kwargs)


class TwoConvTranspose(ConvTranspose):
    """Transposed convolution with kernel 2"""

    def __init__(self, in_channels, out_channels, **kwargs):
        """Initliaze
        
        Args:
            in_channels (int): The number of input channels
            out_channels (int): The number of output channels
            kwargs (dict): See other settings in PyTorch doc

        """
        super().__init__(in_channels, out_channels, 2, bias=False, **kwargs)


class ThreeConv(Conv):
    """Convolution with kernel_size == 3 and 'same' padding"""
    def __init__(self, in_channels, out_channels, **kwargs): 
        super().__init__(in_channels, out_channels, 3, padding=1, **kwargs)


class ProjConv(Conv):
    """Convolution with kernel_size == 1"""
    def __init__(self, in_channels, out_channels, **kwargs): 
        super().__init__(in_channels, out_channels, 1, **kwargs)
