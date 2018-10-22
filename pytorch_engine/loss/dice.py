# -*- coding: utf-8 -*-

import torch
from torch.nn.modules.loss import _Loss


def calc_aver_dice(image1, image2, eps=0.001):
    """Calculate average Dice across channels
    
    Args:
        image1, image2 (Tensor): The images to calculate Dice
        eps (float): Small number to prevent division by zero

    Returns:
        dice (float): The average Dice

    """
    dim = (-3, -2, -1)
    intersection = torch.sum(image1 * image2, dim=dim)
    sum1 = torch.sum(image1, dim=dim)
    sum2 = torch.sum(image2, dim=dim)
    dices = 2 * (intersection + eps) / (sum1 + sum2 + eps)
    dice = torch.mean(dices)
    return dice


class DiceLoss(_Loss):
    def forward(self, input, target):
        return 1 - calc_aver_dice(input, target)
