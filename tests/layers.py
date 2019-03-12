#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pytorch_engine import Config
from pytorch_engine.layers import create_activ, create_norm
from pytorch_engine.layers import create_conv, create_proj, create_three_conv
from pytorch_engine.layers import create_two_upsample, create_dropout


config = Config()
config.dim = 3

config.activ = {'name': 'relu'}
assert create_activ().__str__() == 'ReLU()'
config.activ = {'name': 'leaky_relu', 'negative_slope': 0.1}
assert create_activ().__str__() == 'LeakyReLU(negative_slope=0.1)'

config.norm = {'name': 'batch'}
string = ('BatchNorm%dd(100, eps=1e-05, momentum=0.1, affine=True, '
          'track_running_stats=True)' % config.dim)
assert create_norm(100).__str__() == string
config.norm = {'name': 'instance', 'affine': True}
string = ('InstanceNorm%dd(100, eps=1e-05, momentum=0.1, affine=True, '
          'track_running_stats=False)' % config.dim)
assert create_norm(100).__str__() == string

string = ('Conv%dd(128, 256, kernel_size=(5, 5, 5), stride=(2, 2, 2), '
          'padding=(2, 2, 2))' % config.dim)
assert create_conv(128, 256, 5, padding=2, stride=2).__str__() == string
string = ('Conv%dd(64, 128, kernel_size=(1, 1, 1), '
          'stride=(1, 1, 1))' % config.dim)
assert create_proj(64, 128).__str__() == string
string = ('Conv%dd(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), '
          'padding=(1, 1, 1))' % config.dim)
assert create_three_conv(64, 128).__str__() == string

config.upsample = {'name': 'nearest', 'align_corners': True}
string = 'Interpolate(scale_factor=2, mode=nearest, align_corners=True)'
assert create_two_upsample().__str__() == string
config.upsample = {'name': 'linear', 'align_corners': False}
mode = 'bilinear' if config.dim == 2 else 'trilinear'
string = 'Interpolate(scale_factor=2, mode=%s, align_corners=False)' % mode
assert create_two_upsample().__str__() == string
