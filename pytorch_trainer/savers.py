#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import nibabel as nib
from .observer import Observer
from .config import Config


class Saver(Observer):
    """An abstract class to save the training progress.

    Attributes:
        saving_path_prefix (str): The saving filename prefix. It contains
            slashes if saving to a folder.
    
    """
    def __init__(self, saving_path_prefix):
        super().__init__()
        self.saving_path_prefix = saving_path_prefix

    def _create_saving_directory(self):
        """Creates saving directory"""
        dirname = os.path.dirname(self.saving_path_prefix)
        if len(dirname) > 0:
            os.makedirs(dirname, exist_ok=True)


class ModelSaver(Saver):
    """Saves model periodically.

    The saving period is defined with
    :attr:`pytorch_trainer.config.Config.model_period`.
    
    Attributes:
        saving_path_pattern (str): The filename pattern.
        others (dict): The other instances to save.

    """
    def __init__(self, saving_path_prefix, **others):
        super().__init__(saving_path_prefix)
        pattern = 'checkpoint_{epoch}.pt'
        self.saving_path_pattern = self.saving_path_prefix + pattern
        self.others = others

    def update_on_training_start(self):
        self._epoch_pattern = '%%0%dd' % len(str(self.observable.num_epochs))
        self._create_saving_directory()
        if Config.save_epoch_0:
            self._save()

    def update_on_epoch_end(self):
        """Saves the model."""
        if (self.observable.epoch + 1) % Config.model_period == 0:
            self._save()

    def _save(self):
        """Saves contens."""
        epoch = self.observable.epoch + 1
        epoch = self._epoch_pattern % epoch
        filepath = self.saving_path_pattern.format(epoch=epoch)
        contents = self._get_saving_contents()
        torch.save(contents, filepath)

    def _get_saving_contents(self):
        losses = {k: v.mean for k, v in self.observable.losses.items()}
        contents = {'epoch': self.observable.epoch, 'loss': losses,
                    'trainer_config': Config.save_dict()}
        for name, model in self.observable.models.items():
            contents[name] = model.state_dict()
        contents['optim'] = self.observable.optim.state_dict()
        contents.update(self.others)
        return contents


class PredictionSaver(Saver):
    """Abstract class to save the predictions.
    
    The saving period is defined with
    :attr:`pytorch_trainer.config.Config.pred_period`.

    Attributes:
        subdir (str): The subdirectory to save the current batch.
    
    """
    def update_on_training_start(self):
        self._epoch_pattern = '%%0%dd' % len(str(self.observable.num_epochs))
        self._batch_pattern = '%%0%dd' % len(str(self.observable.num_batches))
        self._create_saving_directory()

    def update_on_epoch_start(self):
        if (self.observable.epoch + 1) % Config.pred_period == 0:
            epoch = self._epoch_pattern % (self.observable.epoch + 1)
            self.subdir = '_'.join([self.saving_path_prefix, epoch])
            os.makedirs(self.subdir, exist_ok=True)

    def update_on_batch_end(self):
        if (self.observable.epoch + 1) % Config.pred_period == 0:
            self._save_outputs()
            self._save_inputs()
            self._save_truths()

    def _save_outputs(self):
        """Saves the outputs."""
        raise NotImplementedError

    def _save_truths(self):
        """Saves the truths."""
        raise NotImplementedError

    def _save_inputs(self):
        """Saves the inputs."""
        raise NotImplementedError


class SegPredSaver(PredictionSaver):
    """Saves prediction of segmentations periodically.
    
    """
    def _save_outputs(self):
        self._save(self._convert_outputs(), 'output.nii.gz')

    def _save_truths(self):
        self._save(self.observable.dumps['truth'], 'truth.nii.gz')

    def _save_inputs(self):
        self._save(self.observable.dumps['input'], 'input.nii.gz' )

    def _save(self, contents, suffix):
        """Saves the contents."""
        num_samples_d = len(str(contents.shape[0]))
        num_channels_d = len(str(contents.shape[1]))
        batch_id = self._batch_pattern % (self.observable.batch + 1)
        for sample_id, sample in enumerate(contents):
            sample_id = ('%%0%dd' % num_samples_d) % (sample_id + 1)
            for channel_id, channel in enumerate(sample):
                channel_id = ('%%0%dd' % num_channels_d) % (channel_id + 1)
                filename = [batch_id, sample_id, channel_id, suffix]
                filename = os.path.join(self.subdir, '_'.join(filename))
                obj = nib.Nifti1Image(channel.numpy().astype(float), np.eye(4))
                obj.to_filename(filename)

    def _convert_outputs(self):
        """Converts output segmentations for saving."""
        outputs = self.observable.dumps['output']
        if outputs.shape[1] > 1:
            outputs = torch.argmax(outputs, dim=1, keepdim=True)
        else:
            outputs = torch.sigmoid(outputs)
        return outputs
