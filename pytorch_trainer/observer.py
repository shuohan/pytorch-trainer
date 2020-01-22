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
    """An abstract class to notify registered :class:`Observer` for updates.

    Notes:
        This a minxin class. If a class inherts from multiple parent classes,
        :class:`Observable` should be put in front.

        Any class inheriting from this class should also be a mixin in order to
        use multiple inheritance, i.e., it should implement

        >>> super().__init__(*args, **kwargs)

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
