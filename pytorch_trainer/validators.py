# -*- coding: utf-8 -*-

import torch

from .observer import Observer, Observable
from .buffer import Buffer


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


class BasicValidator(Validator):
    """A basic validator.

    This validator has only one model to validate. This class is used with
    :class:`pytorch_trainer.BasicTrainer`.

    """
    def _validate(self, data):
        """Validates the models.

        Args:
            data (tuple[torch.Tensor]): The first element is the input tensor to
                the model. The second element is the truth output of the model.

        """
        input, truth = data[0], data[1]
        output = self.observable.models['model'](input)
        loss = self.observable.loss_func(output, truth)
        self.losses['model'].append(self._numpy(loss))

        if Config.dump:
            self._dump('input', input)
            self._dump('output', output)
            self._dump('truth', truth)
