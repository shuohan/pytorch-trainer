# -*- coding: utf-8 -*-


class Observer:
    """Get notified by Trainer to update its status
    
    Args:
        trainer (.trainer.Trainer): Object training the network

    """
    def __init__(self):
        self.trainer = None

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
