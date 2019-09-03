#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .observer import Observer
from ..config import Config


class Printer(Observer):
    """Print the training progress to stdout

    The printed message has format:

        epoch 010/200, batch 01/20, loss 0.1801, metric1: 10.1
        ...
        epoch 010/200, batch 20/20, loss 0.1801, metric1: 10.1
        validation, loss 0.1702, metric1 9.4
        ------------------------------------------------------

    Attributes:
        trainer (Trainer): The trainer holding the training progeress

    """
    def __init__(self, prefix='', show_batch=True):
        self.prefix = prefix
        self.show_batch = show_batch

    def _calc_pattern(self, prefix, total_num):
        """Calculte the message pattern for epoch and batch
            
        Args:
            prefix (str): The prefix of the message
            total_num (int): The number of total epochs/batches

        Returns:
            pattern (str): The calculated pattern

        """
        num_digits = len(str(total_num))
        pattern = '%s %%0%dd/%d' % (prefix, num_digits, total_num)
        return pattern

    def _calc_value_pattern(self):
        pattern = '%%s %%.%df' % Config.decimals
        return pattern

    def update_on_batch_end(self):
        """Print the training progress message"""
        if self.show_batch:
            message = [self.prefix]
            ep = self._calc_pattern('epoch', self.observable.num_epochs)
            bp = self._calc_pattern('batch', self.observable.num_batches)
            message.append(ep % (self.observable.epoch + 1))
            message.append(bp % (self.observable.batch + 1))
            for key, value in self.observable.losses.items():
                current = value.current
                if current.size > 1:
                    space = ' ' * (len(', '.join(message)) + 2)
                    sp = self._calc_pattern('sample', current.shape[0])
                    cp = self._calc_pattern('channel', current.shape[1])
                    for sample_id, sample in enumerate(current):
                        for channel_id, channel in enumerate(sample):
                            tmp = message.copy()
                            tmp.append(sp % (sample_id + 1))
                            tmp.append(cp % (channel_id + 1))
                            tmp.append(('%%.%df' % Config.decimals) % channel)
                            print(', '.join(tmp), flush=True)
                # message.append(self._calc_value_pattern() % (key,value.current))
            # for key, value in self.observable.evaluator.results.items():
            #     message.append(self._calc_value_pattern() % (key,value.current))
            # print(', '.join(message), flush=True)

    def update_on_epoch_end(self):
        message = [self.prefix]
        for key, value in self.observable.losses.items():
            message.append(self._calc_value_pattern() % (key, value.mean))
        # for key, value in self.observable.evaluator.results.items():
        #     message.append(self._calc_value_pattern() % (key, value.mean))
        print(', '.join(message), flush=True)
        print('-' * 80, flush=True)
