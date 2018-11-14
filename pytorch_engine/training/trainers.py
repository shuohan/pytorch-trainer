# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from collections import OrderedDict

from .buffer import Buffer
from .evaluator import Evaluator
from .observer import Observable
from ..config import Configuration


class Trainer(Observable):
    """Abstract"""
    def __init__(self, data_loader, num_epochs=100, num_batches=10):
        self.data_loader = data_loader
        self.num_epochs = num_epochs
        self.num_batches = num_batches

        self.use_gpu = torch.cuda.device_count() > 0

        self.models = OrderedDict()
        self.losses = OrderedDict()
        self.evaluator = Evaluator(self.num_batches)

        self._observers = list()

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
            self._notify_observers_on_batch_start()
            self._train_on_batch(input, truth)
            self._notify_observers_on_batch_end()


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
    def __init__(self, model, loss_func, optimizer, data_loader,
                 num_epochs=500, num_batches=20):
        """Initialize
        
        """
        super().__init__(data_loader=data_loader, num_epochs=num_epochs,
                         num_batches=num_batches)
        if self.use_gpu:
            model = model.cuda()
        self.models['model'] = model

        self.loss_func = loss_func
        self.optimizer = optimizer

        self.losses['loss'] = Buffer(self.num_batches)

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
        output = self.models['model'](input)
        loss = self.loss_func(output, truth)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.losses['loss'].append(loss.item())
        self.evaluator.evaluate(output, truth)


class GANTrainer(Trainer):
    def __init__(self, generator, discriminator, pixel_criterion, adv_criterion,
                 generator_optimizer, discriminator_optimizer,
                 data_loader, pixel_lambda=0.9, adv_lambda=0.1,
                 num_epochs=500, num_batches=20):
        """Initialize
        
        """
        super().__init__(data_loader=data_loader, num_epochs=num_epochs,
                         num_batches=num_batches)

        if self.use_gpu:
            generator = generator.cuda()
            discriminator = discriminator.cuda()
        self.models['generator'] = generator
        self.models['discriminator'] = discriminator

        self.pixel_criterion = pixel_criterion
        self.adv_criterion = adv_criterion
        self.pixel_lambda = pixel_lambda
        self.adv_lambda = adv_lambda
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

        self.losses['gen_loss'] = Buffer(self.num_batches)
        self.losses['pixel_loss'] = Buffer(self.num_batches)
        self.losses['adv_loss'] = Buffer(self.num_batches)
        self.losses['dis_loss'] = Buffer(self.num_batches)

    def _train_on_batch(self, source, target):
        """Train the model for each batch
        
        Args:
            input (torch.Tensor): The input tensor to the model
            truth (torch.Tensor): The target/truth of the output of the model

        """
        source = source.float()
        target = target.float()
        if self.use_gpu:
            source = source.cuda()
            target = target.cuda()

        self.generator_optimizer.zero_grad()
        gen_pred = self.models['generator'](source)
        fake_pred = self.models['discriminator'](gen_pred, source)

        zeros = torch.zeros_like(fake_pred, requires_grad=False)
        ones = torch.ones_like(fake_pred, requires_grad=False)

        adv_loss = self.adv_criterion(fake_pred, ones)
        pixel_loss = self.pixel_criterion(gen_pred, target)
        gen_loss = self.adv_lambda * adv_loss + self.pixel_lambda * pixel_loss
        gen_loss.backward()
        self.generator_optimizer.step()

        self.discriminator_optimizer.zero_grad()
        real_pred = self.models['discriminator'](target, source)
        fake_pred = self.models['discriminator'](gen_pred.detach(), source)
        real_loss = self.adv_criterion(real_pred, ones)
        fake_loss = self.adv_criterion(fake_pred, zeros)
        dis_loss = 0.5 * (fake_loss + real_loss)
        dis_loss.backward()
        self.discriminator_optimizer.step()

        self.losses['gen_loss'].append(gen_loss.item())
        self.losses['pixel_loss'].append(pixel_loss.item())
        self.losses['adv_loss'].append(adv_loss.item())
        self.losses['dis_loss'].append(dis_loss.item())

        self.evaluator.evaluate(gen_pred, target)
