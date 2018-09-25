# -*- coding: utf-8 -*-

import os
import pandas as pd


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
          

class ModelSaver:

    def __init__(self, saving_period, saving_path_pattern,
                 save_weights_only=False):
        self.saving_period = saving_period
        self.saving_path_pattern = saving_path_pattern
        self.save_weights_only = save_weights_only

        dirname = os.path.dirname(self.saving_path_pattern)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

    def save(self, epoch, model):
        """Save"""
        if '{epoch}' in self.saving_path_pattern:
            filename = self.saving_path_pattern.format(epoch=epoch)
        else:
            filename = self.saving_path_pattern

        if (epoch + 1) % self.saving_period:
            if self.save_weights_only:
                torch.save(model.state_dict(), filename)
            else:
                torch.save(model, filename)

class Trainer:
    def __init__(self):
        pass


class SimpleTrainer(Trainer):

    def __init__(self, model, loss_func, optimizer, train_loader,
                 num_epochs=500, val_loader=None, logger=None, saver=None,
                 monitors=dict()):
        """Initialize
        
        """
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.train_loader = train_loader

        self.num_epochs = num_epochs
        self.val_loader = val_loader
        self.logger = logger if logger is not None else Logger()
        self.saver = saver if saver is not None else Saver()
        self.monitors = monitors

        self.use_gpu = torch.cuda.device_count() > 0

        if self.use_gpu:
            self.model.cuda()

    def start(self):
        """Start training

        """
        for epoch in self.num_epochs:
            self._train(epoch)
            self._validate(epoch)
            self.saver.save(epoch, model)
            self.logger.save()

    def _monitor(self, output, truth):
        results = dict()
        for name, func in self.monitors.items():
            results[name] = func(output, truth)
        return results

    def _train(self, epoch):
        for batch, (input, truth) in enumerate(self.train_loader):
            if self.use_gpu:
                input.cuda()
                truth.cuda()
            output = self.model(input)
            loss = self.loss_func(output, truth)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            monitored = self._monitor(output, truth)
            monitored['loss'] = loss.item()
            self.logger.log(epoch, monitored, batch=batch)

    def _validate(self, epoch):
        validation = list()
        with torch.no_grad():
            for input, truth in self.val_loader:
                if self.use_gpu:
                    input.cuda()
                    truth.cuda()
                output = self.model(input)
                loss = self.loss_func(output, truth)
                monitored = self._monitor(output, truth)
                monitored['loss'] = loss.item() 
                validation.append(monitored)
        # self.logger.log(epoch, validation, validation=True)
