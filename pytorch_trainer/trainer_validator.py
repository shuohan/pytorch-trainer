# -*- coding: utf-8 -*-

import torch

from .buffer import Buffer
from .observer import Observable, Observer
from .config import Config


class _TraVal(Observable):
    """Observable for :class:`Trainer` and :class:`Validator`.

    Attributes:
        data_loader (torch.utils.data.DataLoader): The data loader.
        num_batches (int): The number of mini-batches.
        models (dict[torch.nn.Module): The models to train or validate.
        losses (dict[Buffer]): The calculated losses.
        epoch (int): The current epoch.
        batch (int): The current mini-batch.
        dumps (dict[numpy.ndarray]): The intermediate results to dump into cpu.
            It should store ``None`` if not :attr:`pytorch_trainer.Config.dump`.

    Notes:
        This class is a mixin. See :class:`pytorch_trainer.observer.Observable`
        for details.
    
    """
    def __init__(self, data_loader, *args, num_epochs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)
        self.models = dict()
        self.losses = dict()
        self.epoch = 0
        self.batch = 0
        self.dumps = dict()

    def _move_models_to_cuda(self):
        """Moves all models into cuda."""
        for model in self.models.values():
            model.cuda()

    def _move_models_to_cuda(self):
        """Moves all models into cpu."""
        for model in self.models.values():
            model.cpu()

    def _transfer(self, data):
        """Transfers data into GPU or CPU according to config.

        Args:
            data (tuple or torch.Tensor): The data to transfer.

        Returns:
            tuple or torch.Tensor: The transferred data.

        """
        return self._cuda(data) if Config.use_gpu else self._cpu(data)

    def _cuda(self, data):
        """Transfers data into GPU.

        Args:
            data (tuple or torch.Tensor): The data to transfer.

        Returns:
            tuple or torch.Tensor: The transferred data.

        """
        if type(data) is tuple:
            return tuple(self._cuda(d) for d in data)
        elif isinstance(data, torch.Tensor):
            return data.cuda()

    def _cpu(self, data):
        """Dumps data into CPU.

        Args:
            data (tuple or torch.Tensor): The data to dump.

        Returns:
            tuple or torch.Tensor: The dumped data.

        """
        if type(data) is tuple:
            return tuple(self._cpu(d) for d in data)
        elif isinstance(data, torch.Tensor):
            return data.cpu()

    def _numpy(self, data):
        """Converts :class:`torch.Tensor` to :class:`numpy.ndarray'.

        Args:
            data (torch.Tensor): The tensor to convert.

        Returns:
            numpy.ndarray: The converted.

        """
        return data.detach().cpu().numpy()

    def _dump(self, name, data):
        """Dumps data into :attr:`dumps` in CPU as :class:`numpy.ndarray`.

        Args:
            name (str): The name of the data, used as key in :attr:`dumps`.
            data (torch.Tensor): The intermediate data to dump.
        """
        self.dumps[name] = self._numpy(data)


class Trainer(_TraVal):
    """Abstract class for model training.

    """
    def train(self):
        """Trains the models."""
        self._notify_observers_on_training_start()
        self._move_models_to_cuda()
        for self.epoch in range(Config.num_epochs):
            self._notify_observers_on_epoch_start()
            self._set_models_to_train()
            self._train_on_epoch()
            self._notify_observers_on_epoch_end()
        self._notify_observers_on_training_end()

    def _set_models_to_train(self):
        """Sets all modelds to train."""
        for model in self.models.values():
            model.train()

    def _train_on_epoch(self):
        """Trains the models for each epoch."""
        for self.batch, data in enumerate(self.data_loader):
            data = self._transfer(data)
            self._notify_observers_on_batch_start()
            self._train_on_batch(data)
            self._notify_observers_on_batch_end()

    def _train_on_batch(self, data):
        """Trains the models for each batch.

        Args:
            data (tuple or torch.Tensor): The data used to train the models.

        """
        raise NotImplementedError


class Validator(_TraVal, Observer):
    """Abstract class for model validation.

    """
    def update_on_training_start(self):
        """Initializes loss buffers."""
        for name in self.observable.losses.keys():
            self.losses[name] = Buffer(Config.num_epochs)
        self._notify_observers_on_training_start()

    def update_on_epoch_end(self):
        """Validates the models after each training epoch."""
        self.epoch = self.observable.epoch
        if ((self.epoch + 1) % Config.validation_period) == 0:
            with torch.no_grad():
                self._set_models_to_eval()
                for self.batch, data in enumerate(self.data_loader):
                    data = self._transfer(data)
                    self._validate(data)
                    self._notify_observers_on_batch_end()
            self._notify_observers_on_epoch_end()

    def update_on_training_end(self):
        self._notify_observers_on_training_end()

    def _set_models_to_eval(self):
        """Sets all modelds to eval."""
        for model in self.observable.models.values():
            model.eval()

    def _validate(self, data):
        """Validates the models.

        Args:
            data (tuple or torch.Tensor): The data used to validate the models.

        """
        raise NotImplementedError
