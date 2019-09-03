# -*- coding: utf-8 -*-
"""Functions to create other layers.

"""
from ..config import Config


def create_dropout():
    """Creates a spatial dropout layer.

    Note:
        The dropout probability is configured via
        :attr:`pytorch_engine.Config.dropout_rate`, and the spatial dimension is
        configured by :attr:`pytorch_engine.Config.dim`.

    Returns:
        torch.nn.Module: The created dropout layer.

    """
    if Config.dim == 2:
        from torch.nn import Dropout2d
        return Dropout2d(Config.dropout_rate)
    elif Config.dim == 3:
        from torch.nn import Dropout3d
        return Dropout3d(Config.dropout_rate)
