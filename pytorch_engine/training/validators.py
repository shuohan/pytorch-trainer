# -*- coding: utf-8 -*-

import torch

from .observer import Observer, Observable
from .buffer import Buffer


class Validator(Observable, Observer):
    """Validate model

    Attributes:
        data_loader (torch.utils.data.DataLoader): Data loader
        num_epochs (int): The number of epochs
        num_batches (int): The number of batches
        loses (collections.OrderedDict): The losses
        evaluator (.evaluator.Evaluator): Evaluate the model

    """
    def __init__(self, data_loader, saving_period=1):
        """Initialize"""
        super().__init__(data_loader, None)
        self.saving_period = saving_period
        self.output = None

    def update_on_training_start(self):
        """Initialize the losses and evaluator"""
        self.num_epochs = self.observable.num_epochs
        for name in self.observable.losses.keys():
            self.losses[name] = Buffer(self.num_batches)
        self._notify_observers_on_training_start()

    def update_on_epoch_end(self):
        """Validate the model using the models"""
        self.epoch = self.observable.epoch # self.epoch is zero based index
        if ((self.epoch + 1) % self.saving_period) == 0:
            with torch.no_grad():
                for model in self.observable.models.values():
                    model.eval()
                for self.batch, (input, truth) in enumerate(self.data_loader):
                    if self.observable.use_gpu:
                        input, truth = self._cuda(input, truth)
                    self._validate(input, truth)
                    self._notify_observers_on_batch_end()
            self._notify_observers_on_epoch_end()

    def _cuda(self, input, truth):
        return input.cuda(), truth.cuda()

    def update_on_training_end(self):
        self._notify_observers_on_training_end()

    def _validate(self, input, truth):
        """Validate on input and the truth

        Args:
            input (torch.Tensor): The input tensor to the models
            truth (torch.Tensor): The target output

        """
        raise NotImplementedError


class SimpleValidator(Validator):
    """Simple validator with only one model"""
    def _validate(self, input, truth):
        output = self.observable.models['model'](input)
        self.output = output.cpu().detach()
        loss_raw = self.observable.loss_func(output, truth)
        self.losses['loss'].append(loss_raw.cpu().detach().numpy())
        # self.evaluator.evaluate(output, truth)
