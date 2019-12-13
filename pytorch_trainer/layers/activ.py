# -*- coding: utf-8 -*-
"""Functions to create activation layers.

"""
from ..config import Config


def create_activ():
    """Creates an activation layer.

    Note:
        The parameters are configured in :attr:`pytorch_engine.Config.activ`.

    Returns:
        torch.nn.Module: The created activation layer.

    """
    paras = Config.activ.copy()
    paras.pop('name')
    if Config.activ['name'] == 'relu':
        from torch.nn import ReLU
        return ReLU(**paras)
    elif Config.activ['name'] == 'leaky_relu':
        from torch.nn import LeakyReLU
        return LeakyReLU(**paras)
