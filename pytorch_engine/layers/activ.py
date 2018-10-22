#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.nn import LeakyReLU, PReLU, RReLU, ReLU

from ..config import Configuration


config = Configuration()

if config.activ == 'relu':
    Activation_ = ReLU
elif config.activ == 'leaky_relu':
    Activation_ = LeakyReLU
elif config.activ == 'p_relu':
    Activation_ = PReLU
elif config.activ == 'r_relu':
    Activation_ = RReLU


class Activation(Activation_):
    """Customized activation layer"""
    def __init__(self):
        super().__init__(**config.activ_para)
