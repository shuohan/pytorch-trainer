# -*- coding: utf-8 -*-

"""MVC design pattern:

Model (M): The training status
View (V): The logger, printer, model saver, etc.
Controller (C): The training loop

"""

class Status:
    """Abstract"""
    def __init__(self):
        raise NotImplementedError


class Trainer:
    """Abstract"""
    def __init__(self):
        raise NotImplementedError


class Observer:
    """Get notified by Trainer to update its status

    """
    def __init__(self, training_status):
        self.trainig_status = training_status

    def update_training_start(self):
        """Update just before the training starts"""
        pass

    def update_epoch_start(self):
        """Update just before the current epoch starts"""
        pass

    def update_batch_start(self):
        """Update just before the current batch starts"""
        pass

    def update_batch_end(self):
        """Update right after the current batch ends"""
        pass

    def update_epoch_end(self):
        """Update right after the current epoch ends"""
        pass

    def update_training_end(self):
        """Update right after the training ends"""
        pass
