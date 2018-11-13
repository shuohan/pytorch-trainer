#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import os
from .observer import Observer


class ModelSaver(Observer):
    """Save model at certain epochs
    
    Attributes:
        saving_period (int): Save the model every this number of epochs
        saving_path_pattern (str): The filename pattern; "{epoch}" will be
            replaced by the current epoch id
        save_weights_only (bool): Only save the weights
        trainer (.trainers.Trainer): The trainer

    """
    def __init__(self, saving_period, saving_path_prefix,
                 save_weights_only=False):
        """Initialize

        """
        super().__init__()
        self.saving_period = saving_period
        self.save_weights_only = save_weights_only

        self.saving_path_pattern = saving_path_prefix + '{name}_{epoch}'
        if self.save_weights_only:
            self.saving_path_pattern += '_weights.pt'
        else:
            self.saving_path_pattern += '.pt'

        dirname = os.path.dirname(self.saving_path_pattern)
        if dirname and not os.path.isdir(dirname):
            os.makedirs(dirname)

    def update_on_training_start(self):
        """Calculate the number of the digits of the total number of epochs"""
        self._num_digits = len(str(self.trainer.num_epochs))

    def update_on_epoch_end(self):
        """Save the model every self.saving_period number of epochs"""
        if (self.trainer.epoch + 1) % self.saving_period == 0:
            epoch = ('%%0%dd' % self._num_digits) % (self.trainer.epoch + 1)
            for name, model in self.trainer.models.items():
                fn = self.saving_path_pattern.format(name=name, epoch=epoch)
                if self.save_weights_only:
                    torch.save(model.state_dict(), fn)
                else:
                    torch.save(model, fn)
