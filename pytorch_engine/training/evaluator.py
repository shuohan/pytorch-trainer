# -*- coding: utf-8 -*-

from collections import OrderedDict

from .buffer import Buffer
from ..loss import calc_aver_dice
from ..config import Configuration


METRIC_FUNCS = {'dice': calc_aver_dice}


class Evaluator:
    """Evaluate the model using metric functions specified in configuration

    Attributes:
        results (collections.OrderedDict): The evaluation results of the currect
            epoch; each (key, value) pair is a specified metric such as Dice
        _funcs (collections.OrderedDict): The metric functions
        _values (collections.OrderedDict): The buffer holding the evaluation
            resluts

    """
    def __init__(self, buffer_length):
        config = Configuration()
        self._funcs = OrderedDict()
        self._values = OrderedDict()
        self.metric_names = list()
        for name in config.metrics:
            self._funcs[name] = METRIC_FUNCS[name]
            self._values[name] = Buffer(buffer_length)
            self.metric_names.append(name)

    def evaluate(self, prediction, truth):
        """Evaluate the model using prediction and the corresponding truth

        Args:
            prediction (torch.Tensor): The prediction of the model
            truth (torch.Tensor): The corresponding truth

        """
        for key in self._funcs.keys():
            self._values[key].append(self._funcs[key](prediction, truth))

    @property
    def results(self):
        """Get the evaluated results

        Returns:
            results (collections.OrderedDict): The mean evaluated resutls over
                a epoch
        
        """
        results = OrderedDict()
        for key, value in self._values.items():
            results[key] = value.mean
        return results
