# -*- coding: utf-8 -*-

import torch
from torch.nn import Module, Sequential

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


class EncodingBlock(Module):
    """Original UNet encoding block
    
    Attributes:
        in_channels (int): The number of input channels/features
        out_channels (int): The number of output channels/features

    """
    def __init__(self, in_channels, out_channels):
        """Initialize

        Args:
            in_channels (int): The number of input channels/features
            out_channels (int): The number of output channels/features

        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = PostActivConvBlock(in_channels, out_channels)
        self.conv2 = PostActivConvBlock(out_channels, out_channels)

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        return output


class DecodingBlock(Module):
    """Original UNet decoding block with shortcut
    
    Attributes:
        in_channels (int): The number of input channels/features
        shortcut_channels (int): The number of shortcut channels/features
        out_channels (int): The number of output channels/features

    """
    def __init__(self, in_channels, shortcut_channels, out_channels):
        """Initialize

        Args:
            in_channels (int): The number of input channels/features
            shortcut_channels (int): The number of shortcut channels/features
            out_channels (int): The number of output channels/features

        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = Dropout()
        in_channels = in_channels + shortcut_channels
        self.conv1 = PostActivConvBlock(in_channels, out_channels)
        self.conv2 = PostActivConvBlock(out_channels, out_channels)

    def forward(self, input, shortcut):
        output = torch.cat((input, shortcut), dim=1) # concat channels
        output = self.dropout(output)
        output = self.conv1(output)
        output = self.conv2(output)
        return output


class TransDownBlock(Module):
    """2x transition down block
    
    Attributes:
        in_channels (int): The number of input channels/features
        out_channels (int): The number of output channels/features

    """
    def __init__(self, in_channels, out_channels):
        """Initialize

        Args:
            in_channels (int): The number of input channels/features
            out_channels (int): The number of output channels/features

        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pool = Pool(2)
        self.conv1 = PostActivConvBlock(in_channels, out_channels)
        self.conv2 = PostActivConvBlock(out_channels, out_channels)

    def forward(self, input):
        output = self.pool(input)
        output = self.conv1(output)
        output = self.conv2(output)
        return output


class TransUpBlock(Module):
    """2x transition up block
    
    Attributes:
        in_channels (int): The number of input channels/features
        out_channels (int): The number of output channels/features

    """
    def __init__(self, in_channels, out_channels):
        """Initialize

        Args:
            in_channels (int): The number of input channels/features
            out_channels (int): The number of output channels/features

        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = Upsample(scale_factor=2)
        self.conv = ProjConv(in_channels, out_channels)
        self.norm = Normalization(out_channels)
        self.activ = Activation()

    def forward(self, input):
        output = self.up(input)
        output = self.conv(output)
        output = self.norm(output)
        output = self.activ(output)
        return output
