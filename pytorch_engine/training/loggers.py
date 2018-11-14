# -*- coding: utf-8 -*-

import os

from .observer import Observer


class Logger(Observer):
    """Write log
    
    Attributes:
        filename (str): The path to the file to write to
        file (file): The file handler

    """
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        dirname = os.path.dirname(self.filename)
        if dirname and not os.path.isdir(dirname):
            os.makedirs(dirname)

    def update_on_training_start(self):
        self.file = open(self.filename, 'w')
        self._write_header()

    def update_on_training_end(self):
        self.file.flush() 
        self.file.close()

    def _write_header(self):
        """Write the header to the file"""
        raise NotImplementedError


class BasicLogger(Logger):
    """Write training information to file

    """
    def __init__(self, filename, type='training'):
        """Initialize

        Args:
            filename (str): The path to the file to write the info into
            type (str): {"training", "validation"} Select which info to write

        """
        super().__init__(filename)
        if type not in {'training', 'validation'}:
            raise ValueError('type of %s should be "training" or "validation"' \
                             % (self.__class__))
        self.type = type

    def update_on_training_start(self):
        self.losses = self.observable.losses
        self.evaluator = self.observable.evaluator
        super().update_on_training_start()

    def update_on_epoch_end(self):
        line = '%d' % (self.observable.epoch + 1)
        for losses in self.losses.values():
            line += ',%g' % losses.mean
        for value in self.evaluator.results.values():
            line = '%s,%g' % (line, value)
        line += '\n'
        self.file.write(line)
        self.file.flush()

    def _write_header(self):
        header = ['epoch'] + list(self.losses.keys())
        header += self.evaluator.metric_names
        header = ','.join(header) + '\n'
        self.file.write(header)
        self.file.flush()
