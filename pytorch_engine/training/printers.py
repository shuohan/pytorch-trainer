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
        _e_pattern (str): The pattern of the epoch to print
        _b_pattern (str): The pattern of the batch to print

    """
    def __init__(self, trainer):
        """Initialize"""
        self.trainer = trainer
        self._e_pattern = self._calc_pattern('epoch', self.trainer.num_epochs)
        self._b_pattern = self._calc_pattern('batch', self.trainer.num_batches)

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

    def update_training(self):
        """Print the training progress"""
        print(self._get_training_message())

    def update_validation(self):
        print(self._get_validation_message())
        print('-' * 80)

    def _get_training_message(self):
        """Get the training progress message"""
        message = [self._e_pattern % self.trainer.epoch]
        message.append(self._b_pattern % self.trainer.batch)
        for key, value in self.trainer.training_values.items():
            message.append('%s %g' % (key, value.accumulated))
        message = ', '.join(message)
        return message

    def _get_validation_message(self):
        """Get the validation message"""
        message = ['validation']
        for key, value in self.trainer.validation_values.items():
            message.append('%s %g' % (key, value.accumulated))
        message = ', '.join(message)
        return message
