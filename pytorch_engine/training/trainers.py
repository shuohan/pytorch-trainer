# -*- coding: utf-8 -*-

import os
import numpy as np
import torch

from .buffer import Buffer
from .observer import Observable


class Trainer(Observable):
    """Train models

    Attributes:
        data_loader (torch.utils.data.DataLoader): The data loader
        num_epochs (int): The number of epochs
        num_batches (int): The number of batches
        use_gpu (bool): If to use GPU to train or validate

    """
    def train(self):
        """Train the model"""
        self._notify_observers_on_training_start()
        for self.epoch in range(self.num_epochs):
            self._notify_observers_on_epoch_start()
            for model in self.models.values():
                model.train()
            self._train_on_epoch()
            self._notify_observers_on_epoch_end()
        self._notify_observers_on_training_end()

    def _train_on_epoch(self):
        """Train the model for each epoch"""
        for self.batch, (input, truth) in enumerate(self.data_loader):
            if self.use_gpu:
                input, truth = self._cuda(input, truth)
            self._notify_observers_on_batch_start()
            self._train_on_batch(input, truth)
            self._notify_observers_on_batch_end()

    def _cuda(self, input, truth):
        return input.cuda(), truth.cuda()


class SimpleTrainer(Trainer):
    """The Most simple trainer; iterate the training data to update the model

    Attributes:
        loss_func (function): The loss function
        optimizer (torch.optim.Optimizer): The optimizer

    """
    def __init__(self, model, loss_func, optimizer, data_loader,
                 num_epochs=500, num_batches=20):
        """Initialize
        
        """
        super().__init__(data_loader, num_epochs, num_batches)
        self.models['model'] = model if not self.use_gpu else model.cuda()
        self.losses['loss'] = Buffer(self.num_batches)
        self.loss_func = loss_func
        self.optimizer = optimizer

    def _train_on_batch(self, input, truth):
        """Train the model for each batch
        
        Args:
            input (torch.Tensor): The input tensor to the model
            truth (torch.Tensor): The target/truth of the output of the model

        """
        output = self.models['model'](input)
        loss = self.loss_func(output, truth)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.losses['loss'].append(loss.item())
        self.evaluator.evaluate(output, truth)
