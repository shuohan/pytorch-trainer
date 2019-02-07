# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.nn.modules.loss import _Loss

from ..configs import Configurations


configs = Configurations()
if configs.num_dims == 2:
    dim = (-2, -1)
elif configs.num_dims == 3:
    dim = (-3, -2, -1)


def calc_aver_dice(image1, image2, weights=None, eps=0.001):
    """Calculate average Dice across channels
    
    Args:
        image1, image2 (Tensor): The images to calculate Dice
        weights (list): The weight of each label channel; if None, use the same
            weights for all channels
        eps (float): Small number to prevent division by zero

    Returns:
        dice (float): The average Dice

    """
    weights = weights.repeat([image2.shape[0], 1])
    intersection = torch.sum(image1 * image2, dim=dim)
    sum1 = torch.sum(image1, dim=dim)
    sum2 = torch.sum(image2, dim=dim)
    dices = 2 * (intersection + eps) / (sum1 + sum2 + 2 * eps)
    dices = weights * dices if weights is not None else dices
    dice = torch.mean(dices)
    return dice


class DiceLoss(_Loss):
    def __init__(self, weights=None):
        super().__init__()
        if weights is None:
            self.weights = None
        else:
            weights = torch.from_numpy(np.array(weights).astype(np.float32))
            self.weights = weights[None, ...]
            if configs.use_gpu:
                self.weights = self.weights.cuda()

    def forward(self, input, target):
        return 1 - calc_aver_dice(input, target, weights=self.weights)
