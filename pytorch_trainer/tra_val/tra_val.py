# -*- coding: utf-8 -*-

import torch

from ..buffer import Buffer
from ..observer import Observable, Observer
from ..config import Config
from ..funcs import transfer_data_to_cuda, transfer_data_to_cpu
from ..funcs import transfer_models_to_cuda, transfer_models_to_cpu
from ..funcs import set_models_to_train, set_models_to_eval


class _TraVal(Observable):
    """Observable for :class:`Trainer` and :class:`Validator`.

    Attributes:
        data_loader (torch.utils.data.DataLoader): The data loader.
        num_batches (int): The number of mini-batches.
        models (dict[torch.nn.Module): The models to train.
        losses (dict[Buffer]): The calculated losses.
        epoch (int): The current epoch.
        batch (int): The current mini-batch.
        data (torch.Tensor): The current yeilded data.
        dumps (dict[numpy.ndarray]): The intermediate results to dump into cpu.
            It should store ``None`` if not :attr:`pytorch_trainer.Config.dump`.

    Notes:
        This class is a mixin. See :class:`pytorch_trainer.observer.Observable`
        for details.
    
    """
    def __init__(self, data_loader, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)
        self.models = dict()
        self.losses = dict()
        self.epoch = 0
        self.batch = 0
        self.dumps = dict()

    @property
    def metrics(self):
        """Alias of :attr:`losses`."""
        return self.losses

    @property
    def num_epochs(self):
        """Returns number of epochs."""
        return Config.num_epochs

    def _transfer(self, data):
        """Transfers data to cpu or cuda."""
        if Config.use_gpu:
            return transfer_data_to_cuda(data)
        else:
            return transfer_data_to_cpu(data)


class Trainer(_TraVal):
    """Abstract class for model training.

    """
    def train(self):
        """Trains the models."""
        self.notify_observers_on_train_start()
        for self.epoch in range(Config.num_epochs):
            self.notify_observers_on_epoch_start()
            set_models_to_train(self.models)
            self._train_on_epoch()
            self.notify_observers_on_epoch_end()
        self.notify_observers_on_train_end()

    def _train_on_epoch(self):
        """Trains the models for each epoch."""
        for self.batch, self.data in enumerate(self.data_loader):
            self.data = self._transfer(self.data)
            self.notify_observers_on_batch_start()
            self.output = self._train_on_batch(self.data)
            self.notify_observers_on_batch_end()
            self.data = None
            self.output = None

    def _train_on_batch(self, data):
        """Trains the models for each batch.

        Args:
            data (tuple or torch.Tensor): The data used to train the models.

        """
        raise NotImplementedError


class Validator(_TraVal, Observer):
    """Abstract class for model validation.

    """
    def update_on_train_start(self):
        """Initializes loss buffers."""
        for name in self.subject.losses.keys():
            self.models[name] = self.subject.models[name]
            self.losses[name] = Buffer(self.num_batches)
        self.notify_observers_on_train_start()


    def update_on_epoch_end(self):
        """Validates the models after each training epoch."""
        self.epoch = self.subject.epoch
        if ((self.epoch + 1) % Config.val_period) == 0:
            self.notify_observers_on_epoch_start()
            with torch.no_grad():
                set_models_to_eval(self.models)
                for self.batch, self.data in enumerate(self.data_loader):
                    self.data = self._transfer(self.data)
                    self.output = self._validate(self.data)
                    self.notify_observers_on_batch_end()
                    self.data = None
                    self.output = None
            self.notify_observers_on_epoch_end()

    def update_on_train_end(self):
        self.notify_observers_on_train_end()

    def _validate(self, data):
        """Validates the models.

        Args:
            data (tuple or torch.Tensor): The data used to validate the models.

        """
        raise NotImplementedError
