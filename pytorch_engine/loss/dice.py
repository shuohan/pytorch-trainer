# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.nn.modules.loss import _WeightedLoss

from ..funcs import calc_squared_dice_loss, calc_dice_loss, one_hot, prob_encode
from ..config import Config


class SquaredDiceLoss(_WeightedLoss):
    """Wrapper of squared Dice loss"""
    def __init__(self, weight=None):
        super().__init__(weight=weight)
    def forward(self, input, target):
        input = prob_encode(input)
        target_onehot = one_hot(target, input.shape)
        return calc_squared_dice_loss(input, target_onehot, weight=self.weight)


class DiceLoss(_WeightedLoss):
    """Wrapper of Dice loss"""
    def __init__(self, weight=None):
        super().__init__(weight=weight)
    def forward(self, input, target):
        input = prob_encode(input)
        if input.shape[1] > 2:
            target_onehot = one_hot(target, input.shape)
        else:
            target_onehot = target.float()
        average = not Config.eval_separate
        return calc_dice_loss(input, target_onehot, weight=self.weight,
                              average=average)
