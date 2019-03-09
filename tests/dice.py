#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import nibabel as nib

from pytorch_engine.loss import calc_aver_dice


# numeric

input1 = torch.FloatTensor([[0, 0],
                            [0, 0]])
input2 = torch.FloatTensor([[0.8, 0],
                            [0, 0]])
input3 = torch.FloatTensor([[0, 0.7],
                            [0, 0]])
input4 = torch.FloatTensor([[0, 0],
                            [0.6, 0]])
input5 = torch.FloatTensor([[0, 0],
                            [0, 0.5]])
input6 = torch.FloatTensor([[0.8, 0],
                            [0, 0.8]])
input7 = torch.FloatTensor([[0.9, 0.7],
                            [0, 0]])
input8 = torch.FloatTensor([[0, 0.5],
                            [0.6, 0]])
input9 = torch.FloatTensor([[0, 0.8],
                            [0.7, 0.6]])

t1 = torch.cat((input1[None, None, ...],
                input2[None, None, ...],
                input3[None, None, ...]), 1)
t2 = torch.cat((input4[None, None, ...],
                input5[None, None, ...],
                input6[None, None, ...]), 1)
t3 = torch.cat((input7[None, None, ...],
                input8[None, None, ...],
                input9[None, None, ...]), 1)
input = torch.cat((t1, t2, t3), 0)[..., None]

target1 = torch.LongTensor([[0, 2],
                            [1, 2]])
target2 = torch.LongTensor([[2, 2],
                            [0, 1]])
target3 = torch.LongTensor([[1, 0],
                            [2, 2]])
target = torch.cat((target1[None, None, ...],
                    target2[None, None, ...],
                    target3[None, None, ...]), 0)[..., None]

eps = 0.001
dices = ((0 + eps) / (0 + 1 + eps),
         (0 + eps) / (1.8 + eps),
         (1.4 + eps) / (2.7 + eps),
         (1.2 + eps) / (1.6 + eps),
         (1 + eps) / (1.5 + eps),
         (1.6 + eps) / (3.6 + eps),
         (1.4 + eps) / (2.6 + eps),
         (0 + eps) / (2.1 + eps),
         (2.6 + eps) / (4.1 + eps))
dice = calc_aver_dice(input, target, eps=eps)
assert np.round(dice, 4) == np.round(np.mean(dices), 4)

image_path = 'data/AT1021_label.nii.gz'
target = nib.load(image_path).get_data()
labels = np.unique(target)
input = np.zeros([len(labels), *target.shape], dtype=float)
for i, label in enumerate(labels):
    input[i, ...] = target == label
target = np.digitize(target, labels, right=True)

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

target = torch.LongTensor(target[None, None, ...])
input = torch.FloatTensor(input[None, ...])
dice = calc_aver_dice(input, target, eps=eps)
assert np.round(dice, 4) == np.round(np.mean(dices), 4)
