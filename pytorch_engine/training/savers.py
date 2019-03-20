#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from .observer import Observer
from ..config import Config


class ModelSaver(Observer):
    """Save model at certain epochs
    
    Attributes:
        saving_period (int): Save the model every this number of epochs
        saving_path_pattern (str): The filename pattern
        self.others (dict of instances): The other instances to save

    """
    def __init__(self, saving_period, saving_path_prefix, **others):
        """Initialize"""
        super().__init__()
        self.saving_period = saving_period
        self.saving_path_pattern = saving_path_prefix + 'checkpoint_{epoch}.pt'
        self.others = others

    def update_on_training_start(self):
        """Calculate the number of the digits of the total number of epochs"""
        self._num_digits = len(str(self.observable.num_epochs))
        self._create_saving_directory()

    def _create_saving_directory(self):
        """Create saving directory"""
        dirname = os.path.dirname(self.saving_path_pattern)
        if dirname and not os.path.isdir(dirname):
            os.makedirs(dirname)

    def update_on_epoch_end(self):
        """Save the model every self.saving_period number of epochs"""
        if (self.observable.epoch + 1) % self.saving_period == 0:
            filepath = self._get_saving_path()
            contents = self._get_saving_contents()
            torch.save(contents, filepath)

    def _get_saving_path(self):
        epoch = ('%%0%dd' % self._num_digits) % (self.observable.epoch + 1)
        filepath = self.saving_path_pattern.format(epoch=epoch)
        return filepath

    def _get_saving_contents(self):
        contents = {'epoch': self.observable.epoch,
                    'loss': self.observable.losses['loss'].mean,
                    'engine_config': Config()}
        for name, model in self.observable.models.items():
            contents[name] = model.state_dict()
        contents['optimizer'] = self.observable.optimizer.state_dict()
        contents.update(self.others)
        return contents
