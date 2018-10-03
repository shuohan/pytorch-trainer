#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .abstract import Observer


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
    def __init__(self):
        """Initialize"""
        self.training_status = None
        self.validation_status = None

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

    def update_batch_end(self):
        """Print the training progress message"""
        ep = self._calc_pattern('epoch', self.training_status.num_epochs)
        bp = self._calc_pattern('batch', self.training_status.num_batches)
        message = [ep % self.training_status.epoch]
        message.append(bp % self.training_status.batch)
        message.append('loss %g' % self.training_status.loss.accumulated)
        for metric_name, (metric, _) in self.training_status.metrics.items():
            message.append('%s %g' % (metric_name, metric.accumulated))
        print(', '.join(message))

    def update_epoch_end(self):
        message = ['validation']
        message.append('loss %g' % self.validation_status.loss.accumulated)
        for metric_name, (metric, _) in self.validation_status.metrics.items():
            message.append('%s %g' % (metric_name, metric.accumulated))
        print(', '.join(message))
        print('-' * 80)
