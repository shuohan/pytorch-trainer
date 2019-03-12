# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import sigmoid
from torch.nn.functional import softmax
from torch.nn.modules.loss import _WeightedLoss


def calc_aver_dice(input, target, weight=None, eps=0.0001, square=False,
                   activ=False, channel_indices=None):
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
        channel_indices (list): If not None, use these channels for calculation

    Returns:
        dice (torch.FloatTensor): The average Dice

    """
    if activ:
        input = softmax(input, dim=1) if input.shape[1] > 1 else sigmoid(input)

    # one-hot encoding
    target_onehot = torch.FloatTensor(input.shape).zero_()
    target_onehot.scatter_(1, target, 1)

    if channel_indices is not None:
        input = input[:, channel_indices, ...]
        target_onehot = target_onehot[:, channel_indices, ...]

    spatial_dims = tuple(range(2 - len(input.shape), 0))
    intersection = torch.sum(input * target_onehot, dim=spatial_dims)
    if square:
        input = input ** 2
        target_onehot = target_onehot ** 2
    sum1 = torch.sum(input, dim=spatial_dims)
    sum2 = torch.sum(target_onehot, dim=spatial_dims)
    dices = (2 * intersection + eps) / (sum1 + sum2 + eps)

    if weight is not None:
        weight = weight[None, ...].repeat([input.shape[0], 1])
        dices = weight * dices

    dice = torch.mean(dices)

    return dice


class DiceLoss(_WeightedLoss):
    def __init__(self, weight=None):
        super().__init__(weight=weight)
        self.paras = config.loss.copy()
        self.paras.pop('name')

    def forward(self, input, target):
        dice = calc_aver_dice(input, target, weight=self.weight, **self.paras)
        return 1 - dice
