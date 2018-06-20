# -*- coding: utf-8 -*-

from torch.nn import Module

from ..layers import ProjConv, Pool, Upsample
from ..blocks import EncodingBlock as EB
from ..blocks import DecodingBlock as DB
from ..blocks import TransUpBlock as TU
from ..config import Configuration


class UNet(Module):
    """Original UNet with normalization

    https://arxiv.org/pdf/1505.04597.pdf. Always use double convolution blocks
    in pooling and upsampling

    """
    def __init__(self, in_channels, out_classes, num_trans_down,
                 input_conv_channels):
        """Initliaze  

        Args:
            in_channels (int): The number of channels of the input image
            out_classes (int): The number of output labels/classes
            input_conv_channels (int): The number of the output features from
                the input convolution block. It controls the width
            num_trans_down (int): The "level" or the number of pooling. It
                controls the depth of the network.

        """
        super().__init__()
        self.in_channels = in_channels
        self.out_classes = out_classes
        self.num_trans_down = num_trans_down

        # encoding/contracting
        self.eb0 = EB(in_channels, input_conv_channels)
        in_channels = input_conv_channels
        for i in range(self.num_trans_down):
            out_channels = self._calc_out_channels(in_channels)
            setattr(self, 'down%d'%(i+1), Pool(2))
            setattr(self, 'eb%d'%(i+1), EB(in_channels,out_channels))
            in_channels = out_channels

        # decoding/expanding
        for i in range(self.num_trans_down):
            shortcut_ind = self.num_trans_down - i - 1
            out_channels = getattr(self, 'eb%d'%shortcut_ind).out_channels
            setattr(self, 'tu%d'%i, TU(in_channels,out_channels))
            setattr(self, 'db%d'%i, DB(out_channels,out_channels,out_channels))
            in_channels = out_channels
        
        if out_classes == 2:
            out_classes = 1
        self.out = ProjConv(in_channels, out_classes)

    def _calc_out_channels(self, in_channels):
        """Calculate the number of output channels/features

        Args:
            in_channels (int): The number of input channels/features

        Returns:
            out_channels (int): The number of output channels/features

        """
        config = Configuration()
        out_channels = min(in_channels * 2, config.max_channels)
        return out_channels

    def forward(self, input):
        # encoding/contracting
        output = input
        shortcuts = list()
        for i in range(self.num_trans_down+1):
            output = getattr(self, 'eb%d'%i)(output)
            print(output.shape)
            if i < self.num_trans_down:
                shortcuts.insert(0, output)
                output = getattr(self, 'down%d'%(i+1))(output)

        # decoding/expanding
        for i, shortcut in enumerate(shortcuts):
            output = getattr(self, 'tu%d'%i)(output)
            print(output.shape, shortcut.shape)
            output = getattr(self, 'db%d'%i)(output, shortcut)

        output = self.out(output)
        return output