#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import nibabel as nib

from pytorch_engine.loss import create_loss
from pytorch_engine import Config
from pytorch_engine.funcs import calc_dice


loss = create_loss()

# numeric

input111 = torch.FloatTensor([[0.1, 0.2],
                              [0.6, 0.5]])
input112 = torch.FloatTensor([[0.4, 0.6],
                              [0.1, 0.3]])
input121 = torch.FloatTensor([[0.8, 0.1],
                              [0.1, 0.2]])
input122 = torch.FloatTensor([[0.5, 0.2],
                              [0.7, 0.1]])
input131 = torch.FloatTensor([[0.1, 0.7],
                              [0.3, 0.3]])
input132 = torch.FloatTensor([[0.1, 0.2],
                              [0.2, 0.6]])

input211 = torch.FloatTensor([[0.6, 0.1],
                              [0.5, 0.2]])
input212 = torch.FloatTensor([[0.4, 0.6],
                              [0.1, 0.5]])
input221 = torch.FloatTensor([[0.2, 0.4],
                              [0.2, 0.6]])
input222 = torch.FloatTensor([[0.1, 0.2],
                              [0.2, 0.4]])
input231 = torch.FloatTensor([[0.2, 0.5],
                              [0.3, 0.2]])
input232 = torch.FloatTensor([[0.5, 0.2],
                              [0.7, 0.1]])

input11 = torch.cat((input111[..., None], input112[..., None]), 2)
input12 = torch.cat((input121[..., None], input122[..., None]), 2)
input13 = torch.cat((input131[..., None], input132[..., None]), 2)
input1 = torch.cat((input11[None, None, ...],
                    input12[None, None, ...],
                    input13[None, None, ...]), 1)

input21 = torch.cat((input211[..., None], input212[..., None]), 2)
input22 = torch.cat((input221[..., None], input222[..., None]), 2)
input23 = torch.cat((input231[..., None], input232[..., None]), 2)
input2 = torch.cat((input21[None, None, ...],
                    input22[None, None, ...],
                    input23[None, None, ...]), 1)

input = torch.cat((input1, input2), 0)


target11 = torch.LongTensor([[0, 2],
                             [1, 2]])
target12 = torch.LongTensor([[1, 0],
                             [0, 2]])
target1 = torch.cat((target11[..., None], target12[..., None]), 2)

target21 = torch.LongTensor([[1, 0],
                             [0, 0]])
target22 = torch.LongTensor([[2, 1],
                             [0, 1]])
target2 = torch.cat((target21[..., None], target22[..., None]), 2)
target = torch.cat((target1[None, None, ...], target2[None, None, ...]), 0)

eps = Config().eps
dices = ((1.6 + eps) / (5.8 + eps),
         (1.2 + eps) / (4.7 + eps),
         (3.2 + eps) / (5.5 + eps),
         (1.8 + eps) / (7.0 + eps),
         (1.6 + eps) / (5.3 + eps),
         (1 + eps) / (3.7 + eps))
dice = loss(input, target)
assert np.round(dice, 4) == np.round(1 - np.mean(dices), 4)

dice1 = calc_dice(input, target, [0])
dice2 = calc_dice(input, target, [1, 2])
dices1 = ((2.0 + eps) / (6.0 + eps),
          (2.0 + eps) / (8.0 + eps))

dices2 = ((0.0 + eps) / (4.0 + eps),
          (2.0 + eps) / (4.0 + eps),
          (2.0 + eps) / (5.0 + eps),
          (4.0 + eps) / (5.0 + eps))

assert np.round(dice1, 4) == np.round(np.mean(dices1), 4)
assert np.round(dice2, 4) == np.round(np.mean(dices2), 4)

image_path = 'data/AT1021_label.nii.gz'
target = nib.load(image_path).get_data()
labels = np.unique(target)
input = np.zeros([len(labels), *target.shape], dtype=float)
for i, label in enumerate(labels):
    input[i, ...] = target == label
target = np.digitize(target, labels, right=True)
eps = Config().eps
dices = [1]
for i in range(1, input.shape[0]):
    tmp = input[i, ...] 
    indices = np.nonzero(tmp)
    tmp_indices = [None] * 3
    for j in range(len(tmp_indices)):
        tmp_indices[j] = indices[j][:len(indices[j])//2]
    tmp[tuple(tmp_indices)] = 0
    new_size = len(indices[0]) - len(tmp_indices[0]) 
    input[i, ...] = tmp
    d = (2 * new_size + eps) / (len(indices[0]) + new_size + eps)
    dices.append(d)

input = torch.FloatTensor(input[None, ...])
target = torch.LongTensor(target[None, None, ...])
dice = loss(input, target)
assert np.round(dice, 4) == np.round(1 - np.mean(dices), 4)
