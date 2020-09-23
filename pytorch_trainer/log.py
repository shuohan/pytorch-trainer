from pathlib import Path
import warnings
from queue import Queue

from .observer import Observer
from .config import Config


class Writer:
    """Write contents into a .csv file.

    Attributes:
        filename (pathlib.Path): The converted ``filename``.
        fields (iterable[str]): The names of the fields.

    Args:
        filename (str or pathlib.Path): The filename of the output file.

    """
    def __init__(self, filename, fields=[]):
        self.filename = Path(filename)
        self.fields = fields

    def open(self):
        if self.filename.is_file():
            message = 'The file %s exists. Append new contents.' % self.filename
            warnings.warn(message, RuntimeWarning)
        self._file = open(self.filename, 'a')
        self._write_header()

    def _write_header(self):
        header = ','.join(self.fields) + '\n'
        self._file.write(header)

    def write_line(self, data):
        """Writes a line.

        Args:
            data (dict): The contents to write. Its keys should be the same with
                :attr:`fields`.

        """
        values = [self._convert_num(data[k]) for k in self.fields]
        line = ','.join(values) + '\n'
        self._file.write(line)
        self._file.flush()

    def _convert_num(self, num):
        if num.is_integer():
            return '%d' % num
        else:
            return ('%%.%de' % Config().decimals) % num

    def close(self):
        self._file.close()


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
