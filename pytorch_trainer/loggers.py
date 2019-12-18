# -*- coding: utf-8 -*-

import os

from .observer import Observer
from .config import Config


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
    def update_on_training_start(self):
        self.losses = self.observable.losses
        # self.evaluator = self.observable.evaluator
        super().update_on_training_start()

    def update_on_epoch_end(self):
        line_s = '%d' % (self.observable.epoch + 1)
        for value in self.losses.values():
            if Config.eval_separate:
                current = value.all
                for sample_id, sample in enumerate(current):
                    for channel_id, channel in enumerate(sample):
                        line = line_s
                        line += ',%d' % (sample_id + 1)
                        line += ',%d' % (channel_id + 1)
                        line += ',%g' % channel
                        line += '\n'
                        self.file.write(line)
                        self.file.flush()
            else:
                line = line_s + ',%g' % value.mean
                self.file.write(line)
                self.file.flush()
        # for value in self.evaluator.results.values():
        #     line = '%s,%g' % (line, value.mean)

    def _write_header(self):
        if Config.eval_separate:
            header = ['epoch', 'sample', 'channel'] + list(self.losses.keys())
        else:
            header = ['epoch'] + list(self.losses.keys())
        # header += self.evaluator.results.keys()
        header = ','.join(header) + '\n'
        self.file.write(header)
        self.file.flush()
