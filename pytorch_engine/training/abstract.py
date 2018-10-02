#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Trainer:
    """Train networks
    
    Use observer pattern (observable).

    """
    def __init__(self):
        pass


class Observer:
    """Get notified by Trainer to update its status

    """
    def __init__(self, trainer):
        self.trainer = trainer

    def update_training(self):
        raise NotImplementedError

    def update_validation(self):
        raise NotImplementedError
