# -*- coding: utf-8 -*-

import torch
from torch import sigmoid
from torch.nn.functional import softmax
from collections import OrderedDict
from functools import partial

from .buffer import Buffer
from ..loss import calc_aver_dice
from ..configs import Configurations


class MetricFuncs:
    """Return metric functions

    mf = MetricFuncs()
    mf['dice'] # avarege dice across all channels
    mf['dice_0-2_4-6'] # average dice across channels 0, 1, 2, 4, 5, 6

    """
    def __getitem__(self, key):
        if key == 'dice':
            return calc_aver_dice
        elif key.startswith('dice'):
            indices = list()
            for r in key.split('_')[1:]:
                ind = list(map(int, r.split('-')))
                if len(ind) == 2:
                    indices += range(ind[0], ind[1]+1)
                else:
                    indices += ind
            return partial(calc_dice, channel_ids=indices)


def calc_dice(input, target, channel_ids=[0]):
    """Calculate arerage Dice coefficients across specified channels

    Args:
        input, target (torch.Tensor): The images to calculate Dice between
        channel_ids (list): The channels to calculate Dice from

    Returns:
        dice (float): The Dice

    """
    input_tmp = input[:, channel_ids, ...]
    target_tmp = target[:, channel_ids, ...]

    configs = Configurations()
    if configs.num_dims == 2:
        dim = (-2, -1)
        if len(target_tmp.shape) == 4 and target_tmp.shape[1] > 1:
            input_tmp = softmax(input_tmp, dim=1)
        else:
            input_tmp = sigmoid(input_tmp)
    elif configs.num_dims == 3:
        dim = (-3, -2, -1)
        if len(target_tmp.shape) == 5 and target_tmp.shape[1] > 1:
            input_tmp = softmax(input_tmp, dim=1)
        else:
            input_tmp = sigmoid(input_tmp)

    intersection = torch.sum(input_tmp * target_tmp, dim=dim)
    sum1 = torch.sum(input_tmp, dim=dim)
    sum2 = torch.sum(target_tmp, dim=dim)
    dices = 2 * intersection / (sum1 + sum2)
    dice = torch.mean(dices)

    return dice


class Evaluator:
    """Evaluate the model using metric functions specified in configuration

    Attributes:
        results (collections.OrderedDict): The evaluation results of the currect
            epoch; each (key, value) pair is a specified metric such as Dice
        _funcs (collections.OrderedDict): The metric functions
        results (collections.OrderedDict): Hold the evaluation resluts

    """
    def __init__(self, buffer_length):
        configs = Configurations()
        self._funcs = OrderedDict()
        self.results = OrderedDict()
        self.metric_names = list()
        metric_funcs = MetricFuncs()
        for name in configs.metrics:
            self._funcs[name] = metric_funcs[name]
            self.results[name] = Buffer(buffer_length)
            self.metric_names.append(name)

    def evaluate(self, prediction, truth):
        """Evaluate the model using prediction and the corresponding truth

        Args:
            prediction (torch.Tensor): The prediction of the model
            truth (torch.Tensor): The corresponding truth

        """
        for key in self._funcs.keys():
            self.results[key].append(self._funcs[key](prediction, truth))
