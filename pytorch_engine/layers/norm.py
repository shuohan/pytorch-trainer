# -*- coding: utf-8 -*-

from ..configs import Config


def create_norm(num_features):
    config = Config()
    paras = config.norm.copy()
    paras.pop('name')
    if config.dim == 2:
        if config.norm['name'] == 'instance':
            from torch.nn import InstanceNorm2d
            return InstanceNorm2d(num_features, **paras)
        elif config.norm['name'] == 'batch':
            from torch.nn import BatchNorm2d
            return BatchNorm2d(num_features, **paras)
    elif config.dim == 3:
        if config.norm['name'] == 'instance':
            from torch.nn import InstanceNorm3d
            return InstanceNorm3d(num_features, **paras)
        elif config.norm['name'] == 'batch':
            from torch.nn import BatchNorm3d
            return BatchNorm3d(num_features, **paras)
