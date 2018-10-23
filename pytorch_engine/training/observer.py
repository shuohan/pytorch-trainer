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
