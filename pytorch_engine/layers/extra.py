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
    config = Config()
    if config.dim == 2:
        from torch.nn import Dropout2d
        return Dropout2d(config.dropout_rate)
    elif config.dim == 3:
        from torch.nn import Dropout3d
        return Dropout3d(config.dropout_rate)
