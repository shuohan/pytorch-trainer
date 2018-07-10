#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.nn import Dropout3d, Dropout2d

from ..config import Configuration


config = Configuration()

if config.num_dims == 2:
    Dropout_ = Dropout2d
elif config.num_dims == 3:
    Dropout_ = Dropout3d


class Dropout(Dropout_):
    """Customized dropout layer"""

    def __init__(self):
        super().__init__(config.dropout_rate)
