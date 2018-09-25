# -*- coding: utf-8 -*-

import torch
from torch.nn import Module

from ..layers import ProjConv, ThreeConv, Normalization, Activation
from ..layers import Upsample, Dropout


class ResidualEncodingBlock(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = ThreeConv(in_channels, out_channels, stride=2)
        self.norm1 = Normalization(out_channels)
        self.activ1 = Activation()
        self.conv2 = ThreeConv(out_channels, out_channels, stride=1)
        self.norm2 = Normalization(out_channels)
        self.activ2 = Activation()
    def forward(self, input):
        identity = self.conv1(input)
        residue = self.norm1(identity)
        residue = self.activ1(residue)
        residue = self.conv2(residue)
        output = residue + identity
        output = self.norm2(output)
        output = self.activ2(output)
        return output
