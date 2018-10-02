# -*- coding: utf-8 -*-

import torch
from torch.nn import Module

from ..layers import ProjConv, ThreeConv, Normalization, Activation, Pool
from ..layers import Upsample, Dropout


class PostActivConvBlock(Module):
    """Convolution block with post activation
    
    Attributes: 
        in_channels (int): The number of input channels/features
        out_channels (int): The number of output channels/features

    """
    def __init__(self, in_channels, out_channels, stride=1):
        """Initialize 
        Args:
            in_channels (int): The number of input channels/features
            out_channels (int): The number of output channels/features
            stride (int or tuple): The number of convolution strides

        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = ThreeConv(in_channels, out_channels, stride=stride)
        self.norm = Normalization(out_channels)
        self.activ = Activation()

    def forward(self, input):
        output = self.conv(input)
        output = self.norm(output)
        output = self.activ(output)
        return output
