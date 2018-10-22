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
    def __init__(self, saving_period, saving_path_pattern,
                 save_weights_only=False):
        self.saving_period = saving_period
        self.saving_path_pattern = saving_path_pattern
        self.save_weights_only = save_weights_only
        self.trainer = None

        dirname = os.path.dirname(self.saving_path_pattern)
        if dirname and not os.path.isdir(dirname):
            os.makedirs(dirname)

    def update_on_epoch_end(self):
        """Save the model every self.saving_period number of epochs
        
        """
        num_digits = len(str(self.trainer.num_epochs))
        epoch_str = ('%%0%dd' % num_digits) % (self.trainer.epoch + 1)
        if '{epoch}' in self.saving_path_pattern:
            filename = self.saving_path_pattern.format(epoch=epoch_str)
        else:
            filename = self.saving_path_pattern

        if (self.trainer.epoch + 1) % self.saving_period == 0:
            if self.save_weights_only:
                torch.save(self.trainer.model.state_dict(), filename)
            else:
                torch.save(self.trainer.model, filename)
