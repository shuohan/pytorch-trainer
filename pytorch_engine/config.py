# -*- coding: utf-8 -*-

import os
import json
from config import Config_


class Config(Config_):
    """Global configurations

    Attributes:
        dim (int): The number of spatial dimensions (2 or 3)
        kernel_init (str): The kernel weight initialization method
        norm (dict): Parameters of normalization; it has a `name` key to
            determine the type (e.g. batch or instance)
        activ (dict): Parameters of activation; it has a `name` key to determine
            the type (e.g. relu, leaky_relu)
        dropout_rate (float): The rate of dropout
        max_channels (int): The maximum number of feature maps/channels
        upsample (dict): Parameters of upsample; it has a `name` key for the
            type of upsampling mode {'nearest', 'linear'}
        pool (dict): Parameters of pooling; it has key `name` for the type of
            pooling (e.g. 'max', 'avg')
        global_pool (str): The type of globale pooling (max or average)
        decimals (int): The number of decimals of the recorded metrics/loss for
            printing and logging
        metrics (list of str): The metrics used to evaluate the network during
            training and validation
        use_gpu (bool): If True, use GPU; otherwise, use CPU

        _loaded (dict): Loaded configs from .json file
        _attrs (list of str): A list of the configurations for printing

    """
    def __init__(self, config_json=''):
        """Initialize

        Args:
            config_json (str): The path to the json file storing configurations;
                if the file does not exists, use the default settings
                
        """
        super().__init__(config_json)

        self._set_default('dim', 3)
        self._set_default('kernel_init', 'kaiming_normal')
        self._set_default('norm', {'name': 'instance', 'affine': True})
        self._set_default('activ', {'name': 'relu'})
        self._set_default('dropout_rate', 0.2)
        self._set_default('max_channels', 512)
        self._set_default('upsample', {'name': 'nearest',
                                       'align_corners': False})
        self._set_default('pool', {'name', 'max'})
        self._set_default('global_pool', 'avg')
        self._set_default('decimals', 4)
        self._set_default('metrics', list())
        self._set_default('loss', 'dice')
        self._set_default('use_gpu', True)
        self._set_default('encode_output', True)
        self._set_default('eps', 1e-8)
