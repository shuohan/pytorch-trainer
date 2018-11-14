#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from collections import OrderedDict

from .observer import Observer, Observable
from .buffer import Buffer
from .evaluator import Evaluator


class Validator(Observable, Observer):
    """Validate model

    """
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def update_on_training_start(self):
        self.losses = OrderedDict()
        for name in self.observable.losses.keys():
            self.losses[name] = Buffer(self.observable.num_batches)
        self.evaluator = Evaluator(self.observable.num_batches)

    def update_on_epoch_end(self):
        with torch.no_grad():
            for model in self.observable.models.values():
                model.eval()
            for input, truth in self.data_loader:
                input = input.float()
                truth = truth.float()
                if self.observable.use_gpu:
                    input = input.cuda()
                    truth = truth.cuda()
                self._validate(input, truth)

    def _validate(self, input, truth):
        raise NotImplementedError


class SimpleValidator(Validator):
    def _validate(self, input, truth):
        output = self.observable.models['model'](input)
        loss = self.observable.loss_func(output, truth).item()
        self.losses['loss'].append(loss)
        self.evaluator.evaluate(output, truth)


class ConditionalGANValidator(Validator):
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
