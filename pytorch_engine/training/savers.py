#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import nibabel as nib
from .observer import Observer
from ..config import Config
from ..funcs import prob_encode


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
        if (self.observable.epoch + 1) % self.saving_period != 0:
            return

        epoch = ('%%0%dd' % self._num_digits) % (self.observable.epoch + 1)
        subdir = self.saving_path_prefix + epoch
        if not os.path.isdir(subdir):
            os.makedirs(subdir)
        if self.observable.output.shape[1] > 1:
            segs = torch.argmax(self.observable.output, dim=1, keepdim=True)
        else:
            segs = prob_encode(self.observable.output)

        inputs = self.observable.input
        truths = self.observable.truth
        for sample_id, seg in enumerate(segs):
            batch = self.observable.batch
            basename = ('%%0%dd' % len(str(self.observable.num_batches))) % (batch+1)
            basename += '_' + ('%%0%dd' % len(str(len(segs)))) % (sample_id+1)

            input_filename = basename + '_input.nii.gz'
            truth_filename = basename + '_truth.nii.gz'
            output_filename = basename + '_output.nii.gz'
            input_filename = os.path.join(subdir, input_filename)
            truth_filename = os.path.join(subdir, truth_filename)
            output_filename = os.path.join(subdir, output_filename)

            input = inputs[sample_id, ...]
            obj = nib.Nifti1Image(input.numpy().astype(float), np.eye(4))
            obj.to_filename(input_filename)

            truth = truths[sample_id, ...]
            obj = nib.Nifti1Image(truth.numpy().astype(float), np.eye(4))
            obj.to_filename(truth_filename)

            obj = nib.Nifti1Image(seg.numpy().astype(float), np.eye(4))
            obj.to_filename(output_filename)

    def update_on_epoch_end(self):
        pass
