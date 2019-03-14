# -*- coding: utf-8 -*-

from .dice import DiceLoss, SquaredDiceLoss
from ..config import Config

def create_loss(**kwargs):
    config = Config()
    if config.loss == 'dice':
        return DiceLoss(**kwargs)
    elif config.loss == 'squared_dice':
        return SquaredDiceLoss(**kwargs)
    else:
        raise RuntimeError
