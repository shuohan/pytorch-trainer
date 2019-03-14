#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ..config import Config


def create_dropout():
    config = Config()
    if config.dim == 2:
        from torch.nn import Dropout2d
        return Dropout2d(config.dropout_rate)
    elif config.dim == 3:
        from torch.nn import Dropout3d
        return Dropout3d(config.dropout_rate)
