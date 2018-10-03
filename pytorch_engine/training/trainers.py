# -*- coding: utf-8 -*-

import os
import numpy as np
import torch

from .abstract import Trainer
from .status import TrainingStatus
from ..config import Configuration


class SimpleTrainer(Trainer):
    """Most simple trainer

    Attributes:
        model (torch.nn.Module): The network to train
        loss_func (function): The loss function
        optimizer (torch.optim.Optimizer): The optimizer
        train_loader (torch.utils.data.DataLoader): Iterate through the training
            data to optimize the network
        val_loader (torch.utils.data.DataLoader): The validation data
        num_epochs (int): The number total epochs
        num_batches (int): The number of mini-batches per epoch
        use_gpu (bool): Use GPU to train if any GPUs are available
        _observers (list): Observers to notify during training

    """
    def __init__(self, model, loss_func, optimizer, train_loader,
                 num_epochs=500, num_batches=20, val_loader=None):
        """Initialize
        
        """
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.num_epochs = num_epochs
        self.num_batches = num_batches

        self._training_status = TrainingStatus(num_epochs, num_batches)
        self._validation_status = TrainingStatus(num_epochs, len(val_loader))

        self.use_gpu = torch.cuda.device_count() > 0
        if self.use_gpu:
            self.model = self.model.cuda()

        self._observers = list()

    def register_observer(self, observer):
        """Register an observer

        Call the observer `update` and `cleanup` to notify the changes
        
        Args:
            observer (.abstract.Observer): The observer to notify

        """
        observer.training_status = self._training_status
        observer.validation_status = self._validation_status
        self._observers.append(observer) 

    def train(self):
        """Start training

        """
        self._notify_observers_training_start()
        for epoch in range(self.num_epochs):
            self._notify_observers_epoch_start()
            self._train()
            self._validate()
            self._notify_observers_epoch_end()
        self._notify_observers_training_end()

    def _train(self):
        for batch, (input, truth) in enumerate(self.train_loader):
            self._notify_observers_batch_start()
            input = input.float()
            truth = truth.float()
            if self.use_gpu:
                input = input.cuda()
                truth = truth.cuda()
            output = self.model(input)
            loss = self.loss_func(output, truth)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self._training_status.update(loss.item(), output, truth)
            self._notify_observers_batch_end()

    def _validate(self):
        with torch.no_grad():
            for input, truth in self.val_loader:
                input = input.float()
                truth = truth.float()
                if self.use_gpu:
                    input = input.cuda()
                    truth = truth.cuda()
                output = self.model(input)
                loss = self.loss_func(output, truth)
                self._validation_status.update(loss.item(), output, truth)

    def _notify_observers_training_start(self):
        for observer in self._observers:
            observer.update_training_start()

    def _notify_observers_epoch_start(self):
        for observer in self._observers:
            observer.update_epoch_start()

    def _notify_observers_batch_start(self):
        for observer in self._observers:
            observer.update_batch_start()

    def _notify_observers_batch_end(self):
        for observer in self._observers:
            observer.update_batch_end()

    def _notify_observers_epoch_end(self):
        for observer in self._observers:
            observer.update_epoch_end()

    def _notify_observers_training_end(self):
        for observer in self._observers:
            observer.update_training_end()
