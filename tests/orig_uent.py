#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torchviz import make_dot

from pytorch_net_engine.models.orig_unet import UNet

unet = UNet(1, 10, 4, 16)
print(unet)

x = torch.randn(1, 1, 64, 64, 64)
y = unet(x)

dot = make_dot(y, params=dict(unet.named_parameters()))
