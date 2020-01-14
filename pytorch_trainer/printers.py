# -*- coding: utf-8 -*-

import numpy as np

from .observer import Observer
from .config import Config


class Printer(Observer):
    """Prints the training progress to stdout.

    Prints the loss/metric of each mini-batch and the mean across mini-batches
    of each epoch.

    Example outputs

        training (prefix), epoch 001/200, batch 01/10, name, 0.01 (metric value)
        training (prefix), epoch 001/200, batch 02/10, name, 0.02 (metric value)
        ...

    Attributes:
        trainer (Trainer): The trainer holding the training progeress

    """
    def __init__(self, prefix=''):
        self.prefix = prefix
        self._line_length = 0

    def update_on_training_start(self):
        """Initializes printing message."""
        self._epoch_pattern = '%%0%dd' % len(str(self.observable.num_epochs))
        self._batch_pattern = '%%0%dd' % len(str(self.observable.num_batches))
        self._num_epochs = self._epoch_pattern % self.observable.num_epochs
        self._num_batches = self._batch_pattern % self.observable.num_batches

    @property
    def _epoch(self):
        """Returns zero-padded epoch index."""
        epoch = self._epoch_pattern % (self.observable.epoch + 1)
        epoch = '/'.join([epoch, self._num_epochs])
        return epoch

    @property
    def _batch(self):
        """Returns zero-padded batch index."""
        batch = self._batch_pattern % (self.observable.batch + 1)
        batch = '/'.join([batch, self._num_batches])
        return batch

    def update_on_batch_end(self):
        """Prints the training progress message for each batch."""
        line = [self.prefix] if len(self.prefix) > 0 else []
        line += ['epoch', self._epoch, 'batch', self._batch]
        for key, value in self.observable.metrics.items():
            pattern = '%%.%df' % Config.decimals
            mean = pattern % np.mean(value.current)
            line.extend([key, mean])
        line = ', '.join(line)
        self._line_length = max(len(line), self._line_length)
        print(line, flush=True)

    def update_on_epoch_end(self):
        """Prints the training progress message for each epoch."""
        line = [self.prefix] if len(self.prefix) > 0 else []
        for key, value in self.observable.metrics.items():
            pattern = '%%.%df' % Config.decimals
            mean = pattern % np.mean(value.mean)
            line.extend([key, mean])
        print(', '.join(line), flush=True)
        print('-' * self._line_length, flush=True)
