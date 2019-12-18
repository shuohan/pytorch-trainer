# -*- coding: utf-8 -*-

import torch

from .buffer import Buffer
from .observer import Observable, Observer
from .config import Config


class Trainer(Observable):
    """Abstract class for model training.

    """
    def train(self):
        """Trains the models."""
        self._notify_observers_on_training_start()
        self._move_models_to_cuda()
        for self.epoch in range(self.num_epochs):
            self._notify_observers_on_epoch_start()
            self._set_models_to_train()
            self._train_on_epoch()
            self._notify_observers_on_epoch_end()
        self._notify_observers_on_training_end()

    def _set_models_to_train(self):
        """Sets all modelds to train."""
        for model in self.models.values():
            model.train()

    def _train_on_epoch(self):
        """Trains the models for each epoch."""
        for self.batch, data in enumerate(self.data_loader):
            data = self._transfer(data)
            self._notify_observers_on_batch_start()
            self._train_on_batch(data)
            self._notify_observers_on_batch_end()

    def _train_on_batch(self, data):
        """Trains the models for each batch.

        Args:
            data (tuple or torch.Tensor): The data used to train the models.

        """
        raise NotImplementedError


class Validator(Observable, Observer):
    """Abstract class for model validation.

    """
    def update_on_training_start(self):
        """Initializes loss buffers."""
        self.num_epochs = self.observable.num_epochs
        for name in self.observable.losses.keys():
            self.losses[name] = Buffer(self.num_batches)
        self._notify_observers_on_training_start()

    def update_on_epoch_end(self):
        """Validates the models after each training epoch."""
        self.epoch = self.observable.epoch
        if ((self.epoch + 1) % Config.validation_period) == 0:
            with torch.no_grad():
                self._set_models_to_eval()
                for self.batch, data in enumerate(self.data_loader):
                    data = self._transfer(data)
                    self._validate(data)
                    self._notify_observers_on_batch_end()
            self._notify_observers_on_epoch_end()

    def update_on_training_end(self):
        self._notify_observers_on_training_end()

    def _set_models_to_eval(self):
        """Sets all modelds to eval."""
        for model in self.observable.models.values():
            model.eval()

    def _validate(self, data):
        """Validates the models.

        Args:
            data (tuple or torch.Tensor): The data used to validate the models.

        """
        raise NotImplementedError
