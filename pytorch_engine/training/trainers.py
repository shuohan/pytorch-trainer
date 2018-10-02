# -*- coding: utf-8 -*-

import os
import pandas as pd


class Trainer:
    def __init__(self):
        pass


class SimpleTrainer(Trainer):

    def __init__(self, model, loss_func, optimizer, train_loader,
                 num_epochs=500, val_loader=None, logger=None, saver=None,
                 monitors=dict()):
        """Initialize
        
        """
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.train_loader = train_loader

        self.num_epochs = num_epochs
        self.val_loader = val_loader
        self.logger = logger if logger is not None else Logger()
        self.saver = saver if saver is not None else Saver()
        self.monitors = monitors

        self.use_gpu = torch.cuda.device_count() > 0

        if self.use_gpu:
            self.model.cuda()

    def start(self):
        """Start training

        """
        for epoch in self.num_epochs:
            self._train(epoch)
            self._validate(epoch)
            self.saver.save(epoch, model)
            self.logger.save()

    def _monitor(self, output, truth):
        results = dict()
        for name, func in self.monitors.items():
            results[name] = func(output, truth)
        return results

    def _train(self, epoch):
        for batch, (input, truth) in enumerate(self.train_loader):
            if self.use_gpu:
                input.cuda()
                truth.cuda()
            output = self.model(input)
            loss = self.loss_func(output, truth)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            monitored = self._monitor(output, truth)
            monitored['loss'] = loss.item()
            self.logger.log(epoch, monitored, batch=batch)

    def _validate(self, epoch):
        validation = list()
        with torch.no_grad():
            for input, truth in self.val_loader:
                if self.use_gpu:
                    input.cuda()
                    truth.cuda()
                output = self.model(input)
                loss = self.loss_func(output, truth)
                monitored = self._monitor(output, truth)
                monitored['loss'] = loss.item() 
                validation.append(monitored)
        # self.logger.log(epoch, validation, validation=True)
