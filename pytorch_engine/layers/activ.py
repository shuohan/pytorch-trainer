#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ..configs import Config


def create_activ():
    config = Config()
    paras = config.activ.copy()
    paras.pop('name')
    if config.activ['name'] == 'relu':
        from torch.nn import ReLU
        return ReLU(**paras)
    elif config.activ['name'] == 'leaky_relu':
        from torch.nn import LeakyReLU
        return LeakyReLU(**paras)
