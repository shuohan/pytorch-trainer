# -*- coding: utf-8 -*-

import os
import numpy as np
import torch

from .buffer import Buffer
from .observer import Observable
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


class BasicTrainer(Trainer):
    """A basic trainer.

    This trainer has only one model to train. Correspondingly, it also only
    accepts only one loss function to update this model. The :attr:`data_loader`
    should yield a :class:`tuple` of input and truth tensors.

    Notes:
        Multiple loss terms should be defined within :attr:`loss_func`.

    Attributes:
        loss_func (function): The loss function.
        optim (torch.optim.Optimizer): The optimizer.

    """
    def __init__(self, model, loss_func, optim, data_loader, num_epochs=500):
        super().__init__(data_loader, num_epochs)
        self.models['model'] = model
        self.losses['model'] = Buffer(self.num_batches)
        self.loss_func = loss_func
        self.optim = optim

    def _train_on_batch(self, data):
        """Trains the model for each batch.
        
        Args:
            data (tuple[torch.Tensor]): The first element is the input tensor to
                the model. The second element is the truth output of the model.

        """
        input, truth = data[0], data[1]
        output = self.models['model'](input)
        loss = self.loss_func(output, truth)
        self.optim.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.losses['model'].append(self._numpy(loss))

        if Config.dump:
            self._dump('input', input)
            self._dump('output', output)
            self._dump('truth', truth)
