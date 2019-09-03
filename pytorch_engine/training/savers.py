#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import nibabel as nib
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
        self.saving_path_prefix = saving_path_prefix
        self.saving_path_pattern = saving_path_prefix + 'checkpoint_{epoch}.pt'
        self.others = others

    def update_on_training_start(self):
        """Calculate the number of the digits of the total number of epochs"""
        self._num_digits = len(str(self.observable.num_epochs))
        self._create_saving_directory()

    def _create_saving_directory(self):
        """Create saving directory"""
        dirname = os.path.dirname(self.saving_path_prefix)
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
                    'engine_config': Config.save_dict()}
        for name, model in self.observable.models.items():
            contents[name] = model.state_dict()
        contents['optimizer'] = self.observable.optimizer.state_dict()
        contents.update(self.others)
        return contents


class PredictionSaver(ModelSaver):
    def __init__(self, saving_period, saving_path_prefix, labels=None):
        super().__init__(saving_period, saving_path_prefix)
        self.labels = labels

    def update_on_batch_end(self):
        epoch = ('%%0%dd' % self._num_digits) % (self.observable.epoch + 1)
        subdir = self.saving_path_prefix + epoch
        if not os.path.isdir(subdir):
            os.makedirs(subdir)
        if self.observable.output.shape[1] > 1:
            segs = torch.argmax(self.observable.output, dim=1, keepdim=True)
        else:
            segs = self.observable.output > 0.5
        for sample_id, seg in enumerate(segs):
            batch = self.observable.batch
            basename = ('%%0%dd' % len(str(self.observable.num_batches))) % batch
            basename += '_' + ('%%0%dd' % len(str(len(segs)))) % sample_id + '.nii.gz'
            filename = os.path.join(subdir, basename)
            obj = nib.Nifti1Image(seg.numpy().astype(np.uint8), np.eye(4))
            obj.to_filename(filename)

    def update_on_epoch_end(self):
        pass
