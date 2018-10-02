# -*- coding: utf-8 -*-

import os
import numpy as np
import torch

from .abstract import Trainer
from ..config import Configuration


class Value:
    """Handle training loss/metrics

    Output the current loss/metrics or the accumulated value (average). Call
    `update` to append a value and `cleanup` to erase all the tracked values
    
    Attributes:
        id (str): The name of the tracked value
        _all (list): All the tracked values

    """
    def __init__(self, id=''):
        config = Configuration()
        self.decimals = config.decimals
        self.id = id
        self._all = list()

    @property
    def current(self):
        """Return the currect value"""
        return np.round(self._all[-1], self.decimals)

    @property
    def accumulated(self):
        """Return the average accumulated value"""
        return np.round(np.mean(self._all), self.decimals)

    def update(self, current):
        """Add the current value"""
        self._all.append(current)

    def cleanup(self):
        """Clean up all tracked values"""
        self._all = list()


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
        epoch (int): The current epoch id
        batch (int): The current batch id
        metric_funcs (dict): The metric functions to monitor during training
        use_gpu (bool): Use GPU to train if any GPUs are available
        _observers (list): Observers to notify during training

    """
    def __init__(self, model, loss_func, optimizer, train_loader,
                 num_epochs=500, num_batches=20, val_loader=None,
                 metric_funcs=dict()):
        """Initialize
        
        """
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self._epoch = 1
        self._batch = 1

        self.metric_funcs = metric_funcs
        self.training_values = dict()
        self.validation_values = dict()
        for key, value in self.metric_funcs.items():
            self.training_values[key] = Value(id=key)
            self.validation_values[key] = Value(id=key)
        self.training_values['loss'] = Value(id='loss')
        self.validation_values['loss'] = Value(id='loss')

        self.use_gpu = torch.cuda.device_count() > 0
        if self.use_gpu:
            self.model = self.model.cuda()

        self._observers = list()

    @property
    def epoch(self):
        """Current epoch"""
        return self._epoch

    @property
    def batch(self):
        """Current batch"""
        return self._batch

    def register_observer(self, observer):
        """Register an observer

        Call the observer `update` and `cleanup` to notify the changes
        
        Args:
            observer (.abstract.Observer): The observer to notify

        """
        self._observers.append(observer) 

    def register_metric(self, metric_name, metric_func):
        self.metric_funcs[metric_name] = metric_func
        self.training_values[metric_name] = Value(id=metric_name)
        self.validation_values[metric_name] = Value(id=metric_name)

    def train(self):
        """Start training

        """
        for epoch in range(self.num_epochs):
            self._epoch = epoch + 1
            self._train_epoch()
            self._validate()
            self._cleanup_values()

    def _notify_observers_training(self):
        for observer in self._observers:
            observer.update_training()

    def _notify_observers_validation(self):
        for observer in self._observers:
            observer.update_validation()

    def _cleanup_values(self):
        for value in self.training_values.values():
            value.cleanup()
        for value in self.validation_values.values():
            value.cleanup()

    def _train_epoch(self):
        for batch, (input, truth) in enumerate(self.train_loader):
            self._batch = batch + 1
            if self.use_gpu:
                input = input.cuda()
                truth = truth.cuda()
            output = self.model(input)
            loss = self.loss_func(output, truth)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.training_values['loss'].update(loss.item())
            for name, func in self.metric_funcs.items():
                self.training_values[name].update(func(output, truth))

            self._notify_observers_training()

    def _validate(self):
        with torch.no_grad():
            for input, truth in self.val_loader:
                if self.use_gpu:
                    input = input.cuda()
                    truth = truth.cuda()
                output = self.model(input)
                loss = self.loss_func(output, truth)

                self.validation_values['loss'].update(loss.item())
                for name, func in self.metric_funcs.items():
                    self.validation_values[name].update(func(output, truth))
            self._notify_observers_validation()
