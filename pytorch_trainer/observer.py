# -*- coding: utf-8 -*-

from .config import Config


class Observer:
    """Gets notified by :class:`Observable` to update its status.
    
    Args:
        observable (Observable): The observable.

    """
    def __init__(self):
        self.observable = None

    def update_on_training_start(self):
        """Update just before the training starts"""
        pass

    def update_on_epoch_start(self):
        """Update just before the current epoch starts"""
        pass

    def update_on_batch_start(self):
        """Update just before the current batch starts"""
        pass

    def update_on_batch_end(self):
        """Update right after the current batch ends"""
        pass

    def update_on_epoch_end(self):
        """Update right after the current epoch ends"""
        pass

    def update_on_training_end(self):
        """Update right after the training ends"""
        pass


class Observable:
    """Notifies registered :class:`Observer` for updates.

    Attributes:
        data_loader (torch.utils.data.DataLoader): The data loader.
        num_batches (int): The number of mini-batches.
        num_epochs (int): The number of epochs.
        models (dict[torch.nn.Module): The models to train or validate.
        losses (dict[Buffer]): The calculated losses.
        epoch (int): The current epoch.
        batch (int): The current mini-batch.
        dumps (dict[numpy.ndarray]): The intermediate results to dump into cpu.
            It should store ``None`` if not :attr:`pytorch_trainer.Config.dump`.

    """
    def __init__(self, data_loader, num_epochs=None):
        """Initialize
        
        """
        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)
        self.num_epochs = num_epochs

        self.models = dict()
        self.losses = dict()
        self.epoch = 0
        self.batch = 0
        self.dumps = dict()

        self._observers = list()

    def register_observer(self, observer):
        """Registers an observer to get notified.

        Args:
            observer (Observer): The observer to register.

        """
        observer.observable = self
        self._observers.append(observer)

    def _notify_observers_on_training_start(self):
        """"Notifies registered observers on the start of the training."""
        for observer in self._observers:
            observer.update_on_training_start()

    def _notify_observers_on_epoch_start(self):
        """"Notifies the observers on the start of each epoch."""
        for observer in self._observers:
            observer.update_on_epoch_start()

    def _notify_observers_on_batch_start(self):
        """"Notifies the observers on the start of each mini-batch."""
        for observer in self._observers:
            observer.update_on_batch_start()

    def _notify_observers_on_batch_end(self):
        """"Notifies the observers on the end of each mini-batch."""
        for observer in self._observers:
            observer.update_on_batch_end()

    def _notify_observers_on_epoch_end(self):
        """"Notifies the observers on the end of each epoch."""
        for observer in self._observers:
            observer.update_on_epoch_end()

    def _notify_observers_on_training_end(self):
        """"Notifies the observers on the end of the training."""
        for observer in self._observers:
            observer.update_on_training_end()

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
