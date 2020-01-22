# -*- coding: utf-8 -*-

import os

from .observer import Observer
from .tra_val import Trainer, Validator
from .config import Config, LoggerFormat


class Writer:
    """Abstract class to write contents into a file.
    
    Attributes:
        logger:

    """
    def __init__(self, logger):
        self.logger = logger

    @property
    def _epoch(self):
        """Returns the current epoch id."""
        return self.logger.observable.epoch + 1

    def write(self, prefix, values):
        """Writes contents to the file.

        Attributes:
            prefix (str): The prefix written before each line.
            values (dict): The values to write.
        
        """
        raise NotImplementedError

    def write_header(self):
        """Writes table header to the file."""
        raise NotImplementedError


class WideWriter(Writer):
    """Writes the mean across a mini-batch in the wide format.

    Example output:

    epoch,batch,value0,value1,value2
    1,1,0.1,0.2,0.3

    """
    def write_line(self):
        line = '%d' % self._epoch
        for value in self.logger.observable.metrics.values():
            mean = ('%%.%df' % Config.decimals) % value.mean
            line = ','.join([line, mean]) + '\n'
            self.logger.file.write(line )
            self.logger.file.flush()

    def write_header(self):
        header = ['epoch'] + list(self.logger.observable.metrics.keys())
        header = ','.join(header) + '\n'
        self.logger.file.write(header)
        self.logger.file.flush()


class LongWriter(Writer):
    """Writes each channel of each sample in a mini-batch in the long format.

    Example output:

    epoch,batch,sample,channel,name,value0,value1
    1,1,1,1,value0,0.2
    1,1,1,1,value1,0.1

    """
    def write_line(self):
        for key, value in self.logger.observable.metrics.items():
            all_values = value.all
            for batch_id, batch in enumerate(all_values):
                for sample_id, sample in enumerate(batch):
                    for channel_id, channel in enumerate(sample):
                        line = '%d,%d' % (self._epoch, batch_id+1)
                        line = line + ',%d,%d' % (sample_id+1, channel_id+1)
                        val = ('%%.%df' % Config.decimals) % channel
                        line = ','.join([line, key, val]) + '\n'
                        self.logger.file.write(line)
                        self.logger.file.flush()

    def write_header(self):
        header = ['epoch', 'batch', 'sample', 'channel', 'name', 'value']
        header = ','.join(header) + '\n'
        self.logger.file.write(header)
        self.logger.file.flush()


class Logger(Observer):
    """Abstract class to log training or validation progress.
    
    Attributes:
        filename (str): The path to the file to write to.

    Raises:
        RuntimeError: :attr:`Config.logger_fmt` is not in :class:`LoggerFormat`.

    """
    def __init__(self, filename):
        super().__init__()

        self.filename = filename
        dirname = os.path.dirname(self.filename)
        if len(dirname) > 0:
            os.makedirs(dirname, exist_ok=True)

        self.file = open(self.filename, 'w')
        if Config.logger_fmt is LoggerFormat.WIDE:
            self._writer = WideWriter(self)
        elif Config.logger_fmt is LoggerFormat.LONG:
            self._writer = LongWriter(self)
        else:
            raise RuntimeError(Config.logger_fmt, 'is not supported.')

    def update_on_training_start(self):
        """Writes the header."""
        self._writer.write_header()

    def update_on_epoch_end(self):
        """"Writes training or validation progress."""
        self._writer.write_line()

    def update_on_training_end(self):
        """Outputs rest of the information and closes the file."""
        self.file.flush() 
        self.file.close()
