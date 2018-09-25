# -*- coding: utf-8 -*-

from torch.nn import Module

from ..layers import ProjConv, Pool, Upsample
from ..blocks import PostActivConvBlock as Input
from ..blocks import ResidualEncodingBlock as EB
from ..config import Configuration


class BboxNet(Module):
    """Residual Bounding box network

    """
    def __init__(self, in_channels=1, input_conv_channels=16, num_trans_down=5,
                 max_channels=None):
        """Initliaze

        Args:
            in_channels (int): The number of channels of the input image
            input_conv_channels (int): The number of the output features from
                the input convolution block. It controls the width
            num_trans_down (int): The "level" or the number of pooling. It
                controls the depth of the network.

        """
        super().__init__()
        self.in_channels = in_channels
        self.num_trans_down = num_trans_down

        config = Configuration()
        if config.num_dims == 2:
            out_classes = 4
        elif config.num_dims == 3:
            out_classes = 6

        if max_channels is None:
            self.max_channels = config.max_channels
        else:
            self.max_channels = max_channels

        self.input = Input(in_channels, input_conv_channels)
        in_channels = input_conv_channels
        for i in range(self.num_trans_down):
            out_channels = self._calc_out_channels(in_channels)
            setattr(self, 'eb%d'%i, EB(in_channels,out_channels))
            in_channels = out_channels

        self.out = ProjConv(in_channels, out_classes)

    def _calc_out_channels(self, in_channels):
        """Calculate the number of output channels/features

        Args:
            in_channels (int): The number of input channels/features

        Returns:
            out_channels (int): The number of output channels/features

        """
        out_channels = min(in_channels * 2, self.max_channels)
        return out_channels

    def forward(self, input):
        output = self.input(input)
        for i in range(self.num_trans_down):
            output = getattr(self, 'eb%d'%i)(output)
        output = self.out(output)
        return output
