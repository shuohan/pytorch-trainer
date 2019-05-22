# -*- coding: utf-8 -*-

import torch
from collections import OrderedDict
from functools import partial

from .buffer import Buffer
from ..config import Config
from ..funcs import calc_dice


class MetricFuncs:
    """Return metric functions

    mf = MetricFuncs()
    mf['dice'] # avarege dice across all channels
    mf['dice_0-2_4-6'] # average dice across channels 0, 1, 2, 4, 5, 6

    """
    def __getitem__(self, key):
        config = Config()
        if key == 'dice':
            return partial(calc_dice, eps=config.eps)
        elif key.startswith('dice'):
            indices = list()
            for r in key.split('_')[1:]:
                ind = list(map(int, r.split('-')))
                if len(ind) == 2:
                    indices += range(ind[0], ind[1]+1)
                else:
                    indices += ind
            return partial(calc_dice, channel_indices=indices, eps=config.eps)


class Evaluator:
    """Evaluate the model using metric functions specified in configuration

    Attributes:
        results (collections.OrderedDict): The evaluation results of the currect
            epoch; each (key, value) pair is a specified metric such as Dice
        _funcs (collections.OrderedDict): The metric functions
        results (collections.OrderedDict): Hold the evaluation resluts

    """
    def __init__(self, buffer_length):
        config = Config()
        self._funcs = OrderedDict()
        self.results = OrderedDict()
        metric_funcs = MetricFuncs()
        for name in config.metrics:
            self._funcs[name] = metric_funcs[name]
            self.results[name] = Buffer(buffer_length)

    def evaluate(self, prediction, truth):
        """Evaluate the model using prediction and the corresponding truth

        Args:
            prediction (torch.Tensor): The prediction of the model
            truth (torch.Tensor): The corresponding truth

        """
        for key in self._funcs.keys():
            self.results[key].append(self._funcs[key](prediction, truth))
