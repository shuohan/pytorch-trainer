# -*- coding: utf-8 -*-

import os

from .observer import Observer
from .config import Config, LoggerFormat


class Writer:
    """Abstract class to write contents into a file.
    
    Attributes:
        file (io.TextIOWrapper): The file to write.

    """
    def __init__(self):
        self.file = None 

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
    pass


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
    """Logs training or validation progress.
    
    Attributes:
        filename (str): The path to the file to write to.

    """
    def __init__(self, filename):
        super().__init__()

        self.filename = filename
        dirname = os.path.dirname(self.filename)
        if dirname and not os.path.isdir(dirname):
            os.makedirs(dirname)
        
        self._writer = None
        if Config.logger_fmt is LoggerFormat.WIDE:
            self._writer = WideWriter()
        elif Config.logger_fmt is LoggerFormat.LONG:
            self._writer = LongWriter()
        
        self._file = None

    def update_on_training_start(self):
        """Opens a file and writes the header."""
        self._file = open(self.filename, 'w')
        self._writer.file = self._file
        self._writer.write_header()

    def update_on_training_end(self):
        """Outputs rest of the information and closes the file."""
        self._file.flush() 
        self._file.close()


class WideLogger(Logger):
    """A logger that writes the mean across min-batches in the wide format.

    An example output is

    """
    def update_on_epoch_end(self):
        line_s = '%d' % (self.observable.epoch + 1)
        for value in self.observable.losses.values():
            mean = ('%%.%df' % Config.decimals) % value.mean
            line = line_s + ',%s' % mean
            self.file.write(line)
            self.file.flush()

    def _write_header(self):
        header = ['epoch'] + list(self.losses.keys())
        header = ','.join(header) + '\n'
        self.file.write(header)
        self.file.flush()


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
