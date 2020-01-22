#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.nn import Module, ModuleList, ModuleDict
from copy import deepcopy

from pytorch_engine.layers import create_activ, create_three_conv, create_norm
from pytorch_engine.init import init_paras


class Block(Module):
    def __init__(self, ):
        super().__init__()
        self.conv = create_three_conv(1, 2)
        self.norm = create_norm(2)
        self.activ = create_activ()

    def forward(self, input):
        output = self.conv(input)
        output = self.norm(output)
        output = self.activ(output)
        return output


class Net(Module):
    def __init__(self):
        super().__init__()
        self.block = Block()
        self.block_list = ModuleList([Block(), Block()])
        self.block_dict = ModuleDict({'b1': Block(), 'b2': Block()})

    def forward(self, input):
        output = self.block(input)
        for block in self.block_list:
            output = block(output)
        for name, block in self.block_dict.items():
            output = block(output)
        return output


net = Net()
print(net)

para_before = {name: deepcopy(p.data) for name, p in net.named_parameters()}
init_paras(net)
para_after = {name: deepcopy(p.data) for name, p in net.named_parameters()}

for name, pb in para_before.items():
    print('=' * 80)
    print(name)
    print(pb)
    print(para_after[name])
