#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from pytorch_trainer.observer import Observer
from pytorch_trainer.config import Config, LoggerFormat
from pytorch_trainer.tra_val import BasicTrainer, BasicValidator
from pytorch_trainer.funcs import transfer_models_to_cuda
from pytorch_trainer.funcs import transfer_models_to_cpu
from pytorch_trainer.loggers import Logger
from pytorch_trainer.printers import Printer
from pytorch_trainer.savers import ModelSaver, SegPredSaver


class NoiseDataset(Dataset):

    def __init__(self, num_data):
        self.num_data = num_data
        self._data = list()
        for i in range(self.num_data):
            y = np.random.rand()
            x = np.random.randn(1, 5, 5) + y
            y = np.ones((2, 5, 5)) * y
            self._data.append((x.astype(np.float32), y.astype(np.float32)))

    def __getitem__(self, ind):
        return self._data[ind]

    def __len__(self):
        return self.num_data


class TrainerInspector(Observer):
    def update_on_training_start(self):
        assert self.subject.losses['model'].max_length == 8
    def update_on_batch_end(self):
        assert self.subject.models['model'].training

class ValidatorInspector(Observer):
    def update_on_training_start(self):
        assert self.subject.losses['model'].max_length == 2
    def update_on_batch_end(self):
        assert not self.subject.models['model'].training


Config.num_epochs = 20
Config.logger_fmt = LoggerFormat.LONG

model = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 3, padding=1, bias=False),
                            torch.nn.BatchNorm2d(2),
                            torch.nn.ReLU(),
                            torch.nn.Dropout2d())
if Config.use_gpu:
    model = transfer_models_to_cuda(model)
else:
    model = transfer_models_to_cpu(model)

tds = NoiseDataset(16)
tdl = DataLoader(tds, batch_size=2, shuffle=True)

vds = NoiseDataset(4)
vdl = DataLoader(vds, batch_size=2, shuffle=True)

loss_func = torch.nn.MSELoss(reduction='none')
optim = torch.optim.SGD(model.parameters(), lr=0.1)

trainer = BasicTrainer(model, loss_func, optim, tdl)
tinspector = TrainerInspector()
tlogger = Logger('tlog.csv')
tprinter = Printer('training')
ms = ModelSaver('models/')
tp = SegPredSaver('models/tra')
trainer.register_observer(tinspector)
trainer.register_observer(tlogger)
trainer.register_observer(tprinter)
trainer.register_observer(ms)
trainer.register_observer(tp)

validator = BasicValidator(vdl)
vinspector = ValidatorInspector()
vlogger = Logger('vlog.csv')
vprinter = Printer('validati')
vp = SegPredSaver('models/val')
validator.register_observer(vinspector)
validator.register_observer(vlogger)
validator.register_observer(vprinter)
validator.register_observer(vp)
trainer.register_observer(validator)

trainer.train()
