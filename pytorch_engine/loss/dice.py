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


def calc_aver_dice(input, target, weight=None, eps=0.001, square=False,
                   activ=False):
    """Calculate average Dice across channels
    
    Args:
        input (torch.Tensor): The multi-channle testing image; each channel
            corresponds to a label
        target (torch.LongTensor): The reference label image / ground truth
            without one-hot encoding. The labels should be "normalized" so
            values are 0 : num_labels
        weight (torch.Tensor): The weight of each label channel; if None, use
            the same weight for all channels
        eps (float): Small number to prevent division by zero
        square (bool): If True, the denominator is the sum of square; otherwise
            is the sum
        activ (bool): If True, apply softmax / sigmoid to the `input` image

    Returns:
        dice (torch.FloatTensor): The average Dice

    """
    if activ:
        input = softmax(input, dim=1) if input.shape[1] > 1 else sigmoid(input)

    # one-hot encoding
    target_onehot = torch.FloatTensor(input.shape).zero_()
    target_onehot.scatter_(1, target, 1)

    spatial_dims = tuple(range(2 - len(input.shape), 0))
    intersection = torch.sum(input * target_onehot, dim=spatial_dims)
    if square:
        input = input ** 2
        target_onehot = target_onehot ** 2
    sum1 = torch.sum(input, dim=spatial_dims)
    sum2 = torch.sum(target_onehot, dim=spatial_dims)
    dices = (2 * intersection + eps) / (sum1 + sum2 + eps)

    if weight is not None:
        weight = weight.repeat([input.shape[0], 1])
        dices = weight * dices

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
