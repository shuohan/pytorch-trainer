# -*- coding: utf-8 -*-

from collections import OrderedDict

from .evaluator import Evaluator


class Observer:
    """Get notified by Observable to update its status
    
    Args:
        observable (.Observable): The observable

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
    """Trainer or Validator

    Attributes:
        data_loader (torch.utils.data.DataLoader): The data loader
        num_epochs (int): The number of epochs
        num_batches (int): The number of batches
        use_gpu (bool): If to use GPU to train or validate
        models (collections.OrderedDict): The models to train or validate
        losses (collections.OrderedDict): Keep track of the losses
        evaluator (.evaluator.Evaluator): Evaluate the models
        _observers (list): The observers to notify

    """
    def __init__(self, data_loader, num_epochs=100, num_batches=10,
                use_gpu=True):
        """Initialize
        
        """
        self.data_loader = data_loader
        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.use_gpu = use_gpu

        self.models = OrderedDict()
        self.losses = OrderedDict()
        self.evaluator = Evaluator(self.num_batches)

        self._observers = list()

    def register_observer(self, observer):
        """Register an observer

        An registered observer will get notified during training

        Args:
            observer (.abstract.Observer): The observer to notify

        """
        observer.observable = self
        self._observers.append(observer)

    def _notify_observers_on_training_start(self):
        """"Notify the observers for changes on the start of the training"""
        for observer in self._observers:
            observer.update_on_training_start()

    def _notify_observers_on_epoch_start(self):
        """"Notify the observers for changes on the start of each epoch"""
        for observer in self._observers:
            observer.update_on_epoch_start()

    def _notify_observers_on_batch_start(self):
        """"Notify the observers for changes on the start of each mini-batch"""
        for observer in self._observers:
            observer.update_on_batch_start()

    def _notify_observers_on_batch_end(self):
        """"Notify the observers for changes on the end of each mini-batch"""
        for observer in self._observers:
            observer.update_on_batch_end()

    def _notify_observers_on_epoch_end(self):
        """"Notify the observers for changes on the end of each epoch"""
        for observer in self._observers:
            observer.update_on_epoch_end()

    def _notify_observers_on_training_end(self):
        """"Notify the observers for changes on the end of the training"""
        for observer in self._observers:
            observer.update_on_training_end()
