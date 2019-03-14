#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import os
from .observer import Observer
from ..config import Config


class ModelSaver(Observer):
    """Save model at certain epochs
    
    Attributes:
        saving_period (int): Save the model every this number of epochs
        saving_path_pattern (str): The filename pattern; "{epoch}" will be
            replaced by the current epoch id
        save_weights_only (bool): Only save the weights
        model_attributes (list): The attributes of the model to save
        trainer (.trainers.Trainer): The trainer

    """
    def __init__(self, saving_period, saving_path_prefix,
                 save_weights_only=False, model_attributes=list()):
        """Initialize

        """
        super().__init__()
        self.saving_period = saving_period
        self.save_weights_only = save_weights_only
        self.model_attributes = model_attributes

        # self.config_path = saving_path_pattern + 'configs.json'
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
        self._num_digits = len(str(self.observable.num_epochs))
        # Configurations().save(self.config_path)

    def update_on_epoch_end(self):
        """Save the model every self.saving_period number of epochs"""
        if (self.observable.epoch + 1) % self.saving_period == 0:
            epoch = ('%%0%dd' % self._num_digits) % (self.observable.epoch + 1)
            for name, model in self.observable.models.items():
                fn = self.saving_path_pattern.format(name=name, epoch=epoch)
                if self.save_weights_only:
                    torch.save(model.state_dict(), fn)
                else:
                    opti = self.observable.optimizer
                    model_attributes = {key: getattr(model, key)
                                        for key in self.model_attributes}
                    configs = vars(Config())
                    contents = {'configs': configs,
                                'epoch': self.observable.epoch,
                                'loss': self.observable.losses['loss'].mean,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': opti.state_dict()}
                    contents.update(model_attributes)
                    torch.save(contents, fn)
