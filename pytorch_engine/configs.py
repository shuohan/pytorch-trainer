# -*- coding: utf-8 -*-

import os
import json
from py_singleton import Singleton


class Configurations(metaclass=Singleton):
    """Global configurations

    Attributes:
        num_dims (int): The number of spatial dimensions (2 or 3)
        kernel_init (str): The kernel weight initialization method
        norm (str): Batch or instance normalization
        activ (str): The name of the activation (relu ect.)
        activ_para (dict): The activation parameters (ex. leaky relu slope)
        dropout_rate (float): The rate of dropout
        max_channels (int): The maximum number of feature maps/channels
        upsample_mode (str): The type of the upsampling (nearest or linear)
        pool (str): The type of pooling (max or average)
        global_pool (str): The type of globale pooling (max or average)
        decimals (int): The number of decimals of the recorded metrics/loss for
            printing and logging
        metrics (list of str): The metrics used to evaluate the network during
            training and validation

        _configs (dict): Storing the values loaded from the json file

    """
    def __init__(self, json_path=''):
        """Initialize

        Args:
            json_path (str): The path to the json file storing configurations;
                if the file does not exists, use the default settings
                
        """
        if os.path.isfile(json_path):
            self._configs = self._load(json_path)
        else:
            self._configs = dict()

        self._set_default('num_dims', 3)
        self._set_default('kernel_init', 'kaiming_normal')
        self._set_default('norm', 'instance')
        self._set_default('activ', 'relu')
        self._set_default('activ_para', dict())
        self._set_default('dropout_rate', 0.2)
        self._set_default('max_channels', 512)
        self._set_default('upsample_mode', 'nearest')
        self._set_default('upsample_align_corners', False)
        self._set_default('pool', 'max')
        self._set_default('global_pool', 'average')
        self._set_default('decimals', 4)
        self._set_default('metrics', list())

    def _set_default(self, key, value):
        """Set default parameter
        
        Args:
            key (str): The name of the variable
            value (anything): The default value of the variable
        
        """
        value = self._configs[key] if key in self._configs else value
        setattr(self, key, value)
        
    def _load(self, json_path):
        """Load default configurations from a .json file"""
        with open(json_path) as json_file:
            return json.load(json_file)
