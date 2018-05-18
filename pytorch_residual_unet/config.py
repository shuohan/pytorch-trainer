# -*- coding: utf-8 -*-

import os
import json
from py_singleton import Singleton


class Configuration(metaclass=Singleton):
    """Handle configurations
    
    If the path to a .json file with configurations is given, the default
    settings are overwritten by this file.

    Attributes:
        num_dims (int): The number of dimensions. Default: 3
        kernel_init (str): The convolution kernel weight initialization method;
            choices {'kaiming_normal', 'kaiming_uniform', 'xavier_normal',
            'xavier_uniform'}. Default: 'kaming_normal'
        norm (str): The type of normalization layers; choices {'batch',
            'instance'}. Default: 'instance'
        activ (str): The type of non-linear activation; choices {'relu',
            'leaky_relu'}. Default: 'leaky_relu'
        leaky_relu_alpha (float): Alpha (negtive slope) of leaky ReLU
            activation. Default: 0.1
        dropout_rate (float): The rate of dropout. Default: 0.2

    """
    def __init__(self, *args, **kwargs):
        """Initliaze

        Load configurations from a .json file if it exists.

        Args:
            kwargs: config_path (str): The path of the configuration path

        """
        self.num_dims = 3
        self.kernel_init = 'kaiming_normal'
        self.norm = 'instance'
        self.activ = 'leaky_relu'
        self.leaky_relu_alpha = 0.1
        self.dropout_rate = 0.2
        if 'config_path' in kwargs and os.path.isfile(kwargs['config_path']):
            self._load_config(kwargs['config_path'])

    def _load_config(self, config_path):
        """Load configs from .json file

        Args:
            config_path (str): Configurations in a .json file

        """
        with open(config_path) as config_file:
            configs = json.load(config_file)
            for key, value in configs.items():
                setattr(self, key, value)
