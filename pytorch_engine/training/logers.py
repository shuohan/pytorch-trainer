# -*- coding: utf-8 -*-

import os


class Logger:
    def __init__(self, num_epochs, num_batches, decimals=4, filename=None):
        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.decimals = decimals
        self.filename = filename
        self._dataframe = None
        self._header = None

        dirname = os.path.dirname(self.filename)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        if os.path.isfile(self.validation_filename):
            self._val_file = open(self.validation_filename, 'a')
        else:
            self._val_file = open(self.validation_filename, 'w')

        self._epoch_pattern = self._get_progress_pattern(self.num_epochs)
        self._batch_pattern = self._get_progress_pattern(self.num_batches)

    def log(self, epoch, contents):
        contents = pd.DataFrame(contents)
        current = contents.mean()
        if batch is not None:
        for key, value in current.iteritems():
            message.append('%s %%.%df' % (key, self.decimals) % value)
        print(message)

    def _get_progress_pattern(self, prefix, total_num):
        num_digits = len(str(total_num))
        pattern = '%s %%0%dd/%d' % (prefix, num_digits, total_num)
        return pattern

    def _convert_to_dataframe(self, contents):
        if type(contents) is not list \
                and type(contents[contents.keys()[0]]) is not list:
            contents = [contents]
        return pd.DataFrame(contents)


class TrainingLogger(Logger):
    def log(self, epoch, batch, contents):
        contents = self._convert_to_dataframe(contents)
        self._dataframe = pd.concat([self._dataframe, contents])
        current = self._dataframe.iloc[-1]

        if self._header is None:
            self._header = ['epoch'] + sorted(list(contents.columns))

        message = [self._epoch_pattern % epoch]
        message.append(self._batch_pattern % epoch)
        for key, value in current.iteritems():
            message.append('%s %%.%df' % (key, self.decimals) % value)
        message = ', '.join(message)
        print(message)

        if batch == self.num_batches - 1:
            current = self._dataframe.mean()
            message = [self._epoch_pattern % epoch]
            for key, value in current.iteritems():
                message.append('%s %%.%df' % (key, self.decimals) % value)
            message = ', '.join(message)
            print(message)
            writing = [epoch] + [current[key] for key in self._header]
            self._file.write(','.join(writing) + '\n')
