#!/usr/bin/env python
# -*- coding: utf-8 -*-

from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from network_utils import TrainingDataFactory as DF
from network_utils import Dataset3dFactory as DSF
from network_utils.bounding_box import BboxFactoryDecorator

from pytorch_nets import Configuration


image_paths = sorted(glob('data/*_image.*'))
mask_paths = sorted(glob('data/*_label.*'))

config = Configuration()
# augmentation = ['none', 'translation', 'rotation', 'deformation', 'flipping']
augmentation = ['none']
data_factory = DF(dim=1, max_angle=20, max_trans=20,
                  get_data_on_the_fly=False, types=augmentation)
data_factory = BboxFactoryDecorator(data_factory)
t_dataset, v_dataset = DSF.create(data_factory, [], image_paths, mask_paths)
dataloader = DataLoader(t_dataset, batch_size=1, shuffle=True, num_workers=1)

slice_id = 150
counter = 0
for image, bbox in dataloader:
    print(image.shape, bbox.shape)
    image = image[0, 0, ...]
    bbox = bbox[0, ...]
    mask = np.zeros_like(image)
    mask[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]] = True
    plt.figure()
    plt.imshow(image[:, slice_id, :], cmap='gray')
    plt.imshow(mask[:, slice_id, :], alpha=0.3)
    counter += 1
    if counter == 2:
        break
plt.show()
