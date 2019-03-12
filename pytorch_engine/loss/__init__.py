# -*- coding: utf-8 -*-

from .dice import calc_aver_dice, DiceLoss
from ..configs import Config

def create_loss(**kwargs):
    config = Config()
    if config.loss['name'] == 'dice':
        return DiceLoss(**kwargs)
