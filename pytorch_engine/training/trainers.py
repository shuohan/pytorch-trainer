# -*- coding: utf-8 -*-

import os
import numpy as np
import torch

from .buffer import Buffer
from .evaluator import Evaluator
from ..config import Configuration


class Trainer:
    """Abstract"""
    def __init__(self):
        raise NotImplementedError


class SimpleTrainer(Trainer):
    """The Most simple trainer; iterate the training data to update the model

    Attributes:
        model (torch.nn.Module): The network to train
        loss_func (function): The loss function
        optimizer (torch.optim.Optimizer): The optimizer
        training_loader (torch.utils.data.DataLoader): Iterate through the
            training data to optimize the network
        validation_loader (torch.utils.data.DataLoader): The validation data
        num_epochs (int): The number total epochs
        num_batches (int): The number of mini-batches per epoch
        use_gpu (bool): Use GPU to train if any GPUs are available
        _observers (list): Observers to notify during training
        training_losses (.buffer.Buffer): Keep track of losses
        validation_losses (.buffer.Buffer): Keep track of losses of validation
            data
        training_evaluator (.evaluator.Evaluator): Evaludate the model using
            the training data
        validation_evaluator (.evaluator.Evaluator): Evaludate the model using
            the validation data

    """
    def __init__(self, model, loss_func, optimizer, training_loader,
                 num_epochs=500, num_batches=20, validation_loader=None):
        """Initialize
        
        """
        self.model = model
        self.use_gpu = torch.cuda.device_count() > 0
        if self.use_gpu:
            self.model = self.model.cuda()

        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.loss_func = loss_func
        self.optimizer = optimizer

        self.training_loader = training_loader
        self.training_losses = Buffer(self.num_batches)
        self.training_evaluator = Evaluator(self.num_batches)

        self.validation_loader = validation_loader
        if self.validation_loader is None:
            self.validation_losses = None
            self.validation_evaluator = None
        else:
            self.validation_losses = Buffer(len(self.validation_loader))
            self.validation_evaluator = Evaluator(len(self.validation_loader))

        self._observers = list()

    def register_observer(self, observer):
        """Register an observer

        An registered observer will get notified during training

        Args:
            observer (.abstract.Observer): The observer to notify

        """
        observer.trainer = self
        self._observers.append(observer) 

    def train(self):
        """Train the model"""
        self._notify_observers_on_training_start()
        for self.epoch in range(self.num_epochs):
            self._notify_observers_on_epoch_start()
            self._train_on_epoch()
            self._validate_on_epoch()
            self._notify_observers_on_epoch_end()
        self._notify_observers_on_training_end()

    def _train_on_epoch(self):
        """Train the model for each epoch"""
        for self.batch, (input, truth) in enumerate(self.training_loader):
            self._notify_observers_on_batch_start()
            self._train_on_batch(input, truth)
            self._notify_observers_on_batch_end()

    def _train_on_batch(self, input, truth):
        """Train the model for each batch
        
        Args:
            input (torch.Tensor): The input tensor to the model
            truth (torch.Tensor): The target/truth of the output of the model

        """
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
        self.training_losses.append(loss.item())
        self.training_evaluator.evaluate(output, truth)

    def _validate_on_epoch(self):
        """Validate the model for each epoch"""
        with torch.no_grad():
            for input, truth in self.validation_loader:
                self._validate_on_batch(input, truth)

    def _validate_on_batch(self, input, truth):
        """Validate the model for each batch

        Args:
            input (torch.Tensor): The input to the model
            truth (torch.Tensor): The truth of the model output

        """
        input = input.float()
        truth = truth.float()
        if self.use_gpu:
            input = input.cuda()
            truth = truth.cuda()
        output = self.model(input)
        self.validation_losses.append(self.loss_func(output, truth).item())
        self.validation_evaluator.evaluate(output, truth)

    def _notify_observers_on_training_start(self):
        """"Notify the observers for changes on the start of the training"""
        for observer in self._observers:
            observer.update_on_training_start()

    def _notify_observers_on_epoch_start(self):
        """"Notify the observers for changes on the start of each epoch"""
        for observer in self._observers:
            observer.update_on_epoch_start()

    def _notify_observers_on_batch_start(self):
        """"Notify the observers for changes on the start of each mini-batch"""
        for observer in self._observers:
            observer.update_on_batch_start()

    def _notify_observers_on_batch_end(self):
        """"Notify the observers for changes on the end of each mini-batch"""
        for observer in self._observers:
            observer.update_on_batch_end()

    def _notify_observers_on_epoch_end(self):
        """"Notify the observers for changes on the end of each epoch"""
        for observer in self._observers:
            observer.update_on_epoch_end()

    def _notify_observers_on_training_end(self):
        """"Notify the observers for changes on the end of the training"""
        for observer in self._observers:
            observer.update_on_training_end()
