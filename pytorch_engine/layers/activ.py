#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.nn import LeakyReLU, PReLU, RReLU, ReLU

from ..configs import Configurations


configs = Configurations()

if configs.activ == 'relu':
    Activation_ = ReLU
elif configs.activ == 'leaky_relu':
    Activation_ = LeakyReLU
elif configs.activ == 'p_relu':
    Activation_ = PReLU
elif configs.activ == 'r_relu':
    Activation_ = RReLU


class Activation(Activation_):
    """Customized activation layer"""
    def __init__(self):
        super().__init__(**configs.activ_para)
