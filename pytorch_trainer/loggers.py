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
    """Writes the contents in the wide format.

    Example output:

    epoch,batch,value0,value1,value2
    1,1,0.1,0.2,0.3

    """
    def write_line(self):
        line_s = '%d' % (self.logger.observable.epoch + 1)
        for value in self.logger.observable.metrics.values():
            mean = ('%%.%df' % Config.decimals) % value.mean
            line = line_s + ',%s' % mean
            self.logger.file.write(line + '\n')
            self.logger.file.flush()

    def write_header(self):
        header = ['epoch'] + list(self.logger.observable.metrics.keys())
        header = ','.join(header) + '\n'
        self.logger.file.write(header)
        self.logger.file.flush()


class LongWriter(Writer):
    """Writes the contents in the long format.

    Example output:

    epoch,batch,name,value
    1,1,value0,0.1
    1,1,value1,0.2
    1,1,value2,0.3

    """
    pass


class Logger(Observer):
    """Abstract class to log training or validation progress.
    
    Attributes:
        filename (str): The path to the file to write to.

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


class SplitLogger(Logger):
    """A logger writes the losses for each sample in a mini-batch.

    """
    def update_on_epoch_end(self):
        line_s = '%d' % (self.observable.epoch + 1)
        for value in self.observable.losses.values():
            batch_losses = value.all
            for sample_id, sample in enumerate(batch_losses):
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


        # if Config.eval_separate:
        #     header = ['epoch', 'sample', 'channel'] + list(self.losses.keys())
        # else:

    def _write_header(self):
        if Config.eval_separate:
            header = ['epoch', 'sample', 'channel'] + list(self.losses.keys())
        else:
            header = ['epoch'] + list(self.losses.keys())
        # header += self.evaluator.results.keys()
        header = ','.join(header) + '\n'
        self.file.write(header)
        self.file.flush()
