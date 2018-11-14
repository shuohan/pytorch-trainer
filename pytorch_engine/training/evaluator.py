# -*- coding: utf-8 -*-

import torch
from collections import OrderedDict
from functools import partial

from .buffer import Buffer
from ..loss import calc_aver_dice
from ..config import Configuration


config = Configuration()
if config.num_dims == 2:
    dim = (-2, -1)
elif config.num_dims == 3:
    dim = (-3, -2, -1)


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


def calc_dice(image1, image2, channel_ids=[0]):
    """Calculate arerage Dice coefficients across specified channels

    Args:
        image1, image2 (torch.Tensor): The images to calculate Dice between
        channel_ids (list): The channels to calculate Dice from

    Returns:
        dice (float): The Dice

    """
    image1_tmp = image1[:, channel_ids, ...]
    image2_tmp = image2[:, channel_ids, ...]
    intersection = torch.sum(image1_tmp * image2_tmp, dim=dim)
    sum1 = torch.sum(image1_tmp, dim=dim)
    sum2 = torch.sum(image2_tmp, dim=dim)
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
        config = Configuration()
        self._funcs = OrderedDict()
        self.results = OrderedDict()
        self.metric_names = list()
        metric_funcs = MetricFuncs()
        for name in config.metrics:
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
