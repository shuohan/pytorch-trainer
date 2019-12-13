# -*- coding: utf-8 -*-
"""Functions to create normalization layers.

"""
from ..config import Config


def create_norm(num_features):
    """Creates a normalization layer.

    Note:
        The normalization is configured via :attr:`pytorch_engine.Config.norm`,
        and the saptial dimension is configured via
        :attr:`pytorch_engine.Config.dim`.

    Args:
        num_features (int): The number of input channels.

    Returns:
        torch.nn.Module: The created normalization layer.

    """
    paras = Config.norm.copy()
    paras.pop('name')
    if Config.dim == 2:
        if Config.norm['name'] == 'instance':
            from torch.nn import InstanceNorm2d
            return InstanceNorm2d(num_features, **paras)
        elif Config.norm['name'] == 'batch':
            from torch.nn import BatchNorm2d
            return BatchNorm2d(num_features, **paras)
        elif Config.norm['name'] == 'group':
            from torch.nn import GroupNorm
            num_groups = paras.pop('num_groups')
            return GroupNorm(num_groups, num_features, **paras)
    elif Config.dim == 3:
        if Config.norm['name'] == 'instance':
            from torch.nn import InstanceNorm3d
            return InstanceNorm3d(num_features, **paras)
        elif Config.norm['name'] == 'batch':
            from torch.nn import BatchNorm3d
            return BatchNorm3d(num_features, **paras)
        elif Config.norm['name'] == 'group':
            from torch.nn import GroupNorm
            num_groups = paras.pop('num_groups')
            return GroupNorm(num_groups, num_features, **paras)
