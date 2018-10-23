#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .observer import Observer


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

    def update_on_batch_end(self):
        """Print the training progress message"""
        ep = self._calc_pattern('epoch', self.trainer.num_epochs)
        bp = self._calc_pattern('batch', self.trainer.num_batches)
        message = [ep % (self.trainer.epoch + 1)]
        message.append(bp % (self.trainer.batch + 1))
        message.append('loss %g' % self.trainer.training_losses.mean)
        for key, value in self.trainer.training_evaluator.results.items():
            message.append('%s %g' % (key, value))
        print(', '.join(message))

    def update_on_epoch_end(self):
        if self.trainer.validation_loader is not None:
            message = ['validation']
            message.append('loss %g' % self.trainer.validation_losses.mean)
            for key, value in self.trainer.validation_evaluator.results.items():
                message.append('%s %g' % (key, value))
            print(', '.join(message))
        print('-' * 80)
