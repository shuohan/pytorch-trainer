# -*- coding: utf-8 -*-

from .dice import DiceLoss, SquaredDiceLoss
from ..config import Config

def create_loss(**kwargs):
    if Config.loss == 'dice':
        return DiceLoss(**kwargs)
    elif Config.loss == 'squared_dice':
        return SquaredDiceLoss(**kwargs)
    else:
        raise RuntimeError
