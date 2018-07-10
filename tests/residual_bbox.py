#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')
import torch
from torchviz import make_dot

from pytorch_nets.models.residual_bbox_net import BboxNet

input_channels = 16
net = BboxNet(input_conv_channels=input_channels)
print(net)

x = torch.randn(1, 1, 64, 64, 64)
y = net(x)
dot = make_dot(y, params=dict(net.named_parameters()))
dot.format = 'png'
dot.render(filename='res_bbox.png')
