# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import sigmoid
from torch.nn.functional import softmax
from torch.nn.modules.loss import _Loss

from ..configs import Configurations


configs = Configurations()
if configs.num_dims == 2:
    dim = (-2, -1)
elif configs.num_dims == 3:
    dim = (-3, -2, -1)


def calc_aver_dice(input, target, weight=None, eps=0.001):
    """Calculate average Dice across channels
    
    Args:
        input, target (torch.Tensor): The images to calculate Dice
        weight (torch.Tensor): The weight of each label channel; if None, use
            the same weight for all channels
        eps (float): Small number to prevent division by zero

    Returns:
        dice (float): The average Dice

    """
    if configs.num_dims == 2:
        if len(target.shape) == 4 and target.shape[1] > 1:
            input = softmax(input, dim=1)
        else:
            input = sigmoid(input)
    elif configs.num_dims == 3:
        if len(target.shape) == 5 and target.shape[1] > 1:
            input = softmax(input, dim=1)
        else:
            input = sigmoid(input)

    weight = weight.repeat([target.shape[0], 1])
    intersection = torch.sum(input * target, dim=dim)
    sum1 = torch.sum(input, dim=dim)
    sum2 = torch.sum(target, dim=dim)
    dices = 2 * (intersection + eps) / (sum1 + sum2 + 2 * eps)
    dices = weight * dices if weight is not None else dices
    dice = torch.mean(dices)
    return dice


class DiceLoss(_Loss):
    def __init__(self, weight=None):
        super().__init__()
        if weight is None:
            self.weight = None
        else:
            weight = torch.from_numpy(np.array(weight).astype(np.float32))
            self.weight = weight[None, ...]
            if configs.use_gpu:
                self.weight = self.weight.cuda()

    def forward(self, input, target):
        return 1 - calc_aver_dice(input, target, weight=self.weight)
