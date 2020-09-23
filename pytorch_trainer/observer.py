"""Observer design pattern.

"""
class Observer:
    """Gets notified by :class:`Subject` to update its status.

    Note:
        This a minxin class. If a class inherts from multiple parent classes,
        this class should be put in front. If all parent class are mixins,
        the order does not matter.

        Any class inheriting from this class should also be a mixin in order to
        use multiple inheritance, i.e., it should implement

        >>> super().__init__(*args, **kwargs)
    
    Attributes:
        subject (Subject): The subject that is been observed.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subject = None

    def update_on_train_start(self):
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

    def update_on_train_end(self):
        """Update right after the training ends"""
        pass


class Subject:
    """An abstract class to notify registered :class:`Observer` for updates.

    Note:
        This a minxin class. If a class inherts from multiple parent classes,
        this class should be put in front. If all parent class are mixins,
        the order does not matter.

        Any class inheriting from this class should also be a mixin in order to
        use multiple inheritance, i.e., it should implement

        >>> super().__init__(*args, **kwargs)

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._observers = list()

    def register(self, observer):
        """Registers an observer to get notified.

        Args:
            observer (Observer): The observer to register.

        """
        observer.subject = self
        self._observers.append(observer)

    def notify_observers_on_train_start(self):
        """"Notifies registered observers on the start of the training."""
        for observer in self._observers:
            observer.update_on_train_start()

    def notify_observers_on_epoch_start(self):
        """"Notifies the observers on the start of each epoch."""
        for observer in self._observers:
            observer.update_on_epoch_start()

    def notify_observers_on_batch_start(self):
        """"Notifies the observers on the start of each mini-batch."""
        for observer in self._observers:
            observer.update_on_batch_start()

    def notify_observers_on_batch_end(self):
        """"Notifies the observers on the end of each mini-batch."""
        for observer in self._observers:
            observer.update_on_batch_end()

    def notify_observers_on_epoch_end(self):
        """"Notifies the observers on the end of each epoch."""
        for observer in self._observers:
            observer.update_on_epoch_end()

    def notify_observers_on_train_end(self):
        """"Notifies the observers on the end of the training."""
        for observer in self._observers:
            observer.update_on_train_end()
