# -*- coding: utf-8 -*-

import torch

from .observer import Observer, Observable
from .buffer import Buffer
from ..config import Config
from ..funcs import prob_encode


class Validator(Observable, Observer):
    """Validate model

    Attributes:
        data_loader (torch.utils.data.DataLoader): Data loader
        num_epochs (int): The number of epochs
        num_batches (int): The number of batches
        loses (collections.OrderedDict): The losses
        evaluator (.evaluator.Evaluator): Evaluate the model

    """
    def __init__(self, data_loader, num_batches=10):
        """Initialize"""
        super().__init__(data_loader, None, num_batches)

    def update_on_training_start(self):
        """Initialize the losses and evaluator"""
        self.num_epochs = self.observable.num_epochs
        for name in self.observable.losses.keys():
            self.losses[name] = Buffer(self.num_batches)
        self._notify_observers_on_training_start()

    def update_on_epoch_end(self):
        """Validate the model using the models"""
        self.epoch = self.observable.epoch
        with torch.no_grad():
            for model in self.observable.models.values():
                model.eval()
            for input, truth in self.data_loader:
                input = input.float()
                truth = truth.long()
                if self.observable.use_gpu:
                    input = input.cuda()
                    truth = truth.cuda()
                self._validate(input, truth)
        self._notify_observers_on_epoch_end()

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
        if Config().encode_output:
            output = prob_encode(output)
        loss = self.observable.loss_func(output, truth).item()
        self.losses['loss'].append(loss)
        self.evaluator.evaluate(output, truth)


class ConditionalGANValidator(Validator):
    """Validate the conditional GAN"""
    def _validate(self, input, truth):
        gen_pred = self.observable.models['generator'](input)
        fake_pred = self.observable.models['discriminator'](gen_pred, input)
        real_pred = self.observable.models['discriminator'](truth, input)

        zeros = torch.zeros_like(fake_pred, requires_grad=False)
        ones = torch.ones_like(fake_pred, requires_grad=False)

        adv_loss = self.observable.adv_criterion(fake_pred, ones).item()
        pixel_loss = self.observable.pixel_criterion(gen_pred, truth).item()
        gen_loss = self.observable.adv_lambda * adv_loss \
                 + self.observable.pixel_lambda * pixel_loss
        fake_loss = self.observable.adv_criterion(fake_pred, zeros).item()
        real_loss = self.observable.adv_criterion(real_pred, ones).item()
        dis_loss = 0.5 * (fake_loss + real_loss)

        self.losses['gen_loss'].append(gen_loss)
        self.losses['pixel_loss'].append(pixel_loss)
        self.losses['adv_loss'].append(adv_loss)
        self.losses['dis_loss'].append(dis_loss)

        self.evaluator.evaluate(gen_pred, truth)
