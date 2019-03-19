# -*- coding: utf-8 -*-

import torch
from torch import sigmoid
from torch.nn.functional import softmax

from .config import Config


def prob_encode(input):
    """Apply softmax or sigmoid

    Args:
        input (torch.Tensor): Input tensor

    Returns:
        result (torch.Tensor): The result

    """
    result = softmax(input, dim=1) if input.shape[1] > 1 else sigmoid(input)
    return result


def one_hot(input, shape):
    """One hot encoding; torch does not have it as the current version

    Args:
        input (torch.LongTensor): The tensor to encode. The values shoule be
            "normalized" to 0 : num_labels 

    Returns:
        result (torch.FloatTensor): The encoded tensor

    """
    result = torch.FloatTensor(shape).zero_()
    if Config().use_gpu:
        result = result.cuda()
    result.scatter_(1, input, 1)
    return result


def _calc_dices(input, target, eps=0.001):
    """Calculate dices for each sample and each channel

    Args:
        input (torch.FloatTensor): The input tensor
        target (torch.FloatTensor): The target tensor, one_hot encoded

    Returns:
        dices (torch.FloatTensor): The dices of each sample (first dim) and each
            channel (second dim)

    """
    spatial_dims = tuple(range(2 - len(input.shape), 0))
    intersection = torch.sum(input * target, dim=spatial_dims)
    sum1 = torch.sum(input, dim=spatial_dims)
    sum2 = torch.sum(target, dim=spatial_dims)
    dices = (2 * intersection + eps) / (sum1 + sum2 + eps)
    return dices


def _calc_squared_dices(input, target, eps=0.001):
    """Calculate squared dices for each sample and each channel

    Args:
        input (torch.FloatTensor): The input tensor
        target (torch.FloatTensor): The target tensor, one_hot encoded
        eps (float): The smoothing term preventing division by 0

    Returns:
        dices (torch.FloatTensor): The dices of each sample (first dim) and each
            channel (second dim)

    """
    spatial_dims = tuple(range(2 - len(input.shape), 0))
    intersection = torch.sum(input * target, dim=spatial_dims)
    sum1 = torch.sum(input ** 2, dim=spatial_dims)
    sum2 = torch.sum(target ** 2, dim=spatial_dims)
    dices = (2 * intersection + eps) / (sum1 + sum2 + eps)
    return dices


def calc_weighted_average(vals, weight):
    """Calculate weighted average along the second dim of values 

    Args:
        vals (torch.Tensor): The values to weight; the first dim is samples
        weight (torch.Tensor): The 1d weights to apply to the second dim of vals

    Returns:
        result (torch.Tensor): The result

    """
    weight = weight[None, ...].repeat([vals.shape[0], 1])
    result = torch.mean(weight * vals)
    return result


def calc_dice_loss(input, target, weight=None):
    """Calculate the dice loss

    Args:
        input (torch.Tensor): The input tensor
        target (torch.Tensor): The target tensor
        eps (float): The smoothing term preventing division by 0

    Return:
        dice (torch.Tensor): The weighted dice

    """
    dices = _calc_dices(input, target, eps=Config().eps)
    if weight is None:
        dice = torch.mean(dices)
    else:
        dice = calc_weighted_average(dices, weight)
    return 1 - dice


def calc_squared_dice_loss(input, target, weight=None):
    """Calculate the squared dice loss

    Args:
        input (torch.Tensor): The input tensor
        target (torch.Tensor): The target tensor
        eps (float): The smoothing term preventing division by 0

    Return:
        dice (torch.Tensor): The weighted dice

    """
    dices = _calc_squared_dices(input, target, eps=Config().eps)
    if weight is None:
        dice = torch.mean(dices)
    else:
        dice = calc_weighted_average(dices, weight)
    return 1 - dice


def calc_dice(input, target, channel_indices=None):
    """Calculate average Dice coefficients across samples and channels

    Args:
        input (torch.Tensor): The input tensor
        target (torch.Tensor): The target tensor
        channel_indices (list of int): The channels to calculate dices across.
            If None, use all channels

    Returns:
        dice (torch.Tensor): The average Dice

    """
    input_seg = one_hot(torch.argmax(input, dim=1, keepdim=True), input.shape)
    target_onehot = one_hot(target, input.shape)
    if channel_indices is not None:
        input_seg = input_seg[:, channel_indices, ...]
        target_onehot = target_onehot[:, channel_indices, ...]
    dices = _calc_dices(input_seg, target_onehot, eps=0)
    return torch.mean(dices)


def count_trainable_paras(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
