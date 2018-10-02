#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.nn import Conv2d, Conv3d, ConvTranspose2d, ConvTranspose3d

from ..config import Configuration


config = Configuration()
if config.num_dims == 2:
    Conv = Conv2d
    ConvTranspose = ConvTranspose2d
elif config.num_dims == 3:
    Conv = Conv3d
    ConvTranspose = ConvTranspose3d


class TwoConvTranspose(ConvTranspose):
    """Transposed convolution with kernel 2"""

    def __init__(self, in_channels, out_channels, **kwargs):
        """Initliaze
        
        Args:
            in_channels (int): The number of input channels
            out_channels (int): The number of output channels
            kwargs (dict): See other settings in PyTorch doc

        """
        super().__init__(in_channels, out_channels, 2, **kwargs)


class ThreeConv(Conv):
    """Convolution with kernel_size == 3 and 'same' padding"""
    def __init__(self, in_channels, out_channels, **kwargs): 
        super().__init__(in_channels, out_channels, 3, padding=1, **kwargs)


class ProjConv(Conv):
    """Convolution with kernel_size == 1"""
    def __init__(self, in_channels, out_channels, **kwargs): 
        super().__init__(in_channels, out_channels, 1, **kwargs)
