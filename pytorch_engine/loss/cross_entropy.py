# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.nn.modules.loss import CrossEntropyLoss as CrossEntropyLoss_

from ..configs import Configurations


configs = Configurations()
if configs.num_dims == 2:
    dim = (-2, -1)
elif configs.num_dims == 3:
    dim = (-3, -2, -1)


class CrossEntropyLoss(CrossEntropyLoss_):
    def forward(self, input, target):
        return super().forward(input, target.max(1)[1])
