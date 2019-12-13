# -*- coding: utf-8 -*-

from config import Config as Config_


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
        loss (str): The type of the loss
        encode_output (bool): If True, apply softmax/sigmoid to the model output
        eps (float): A small number

        _loaded (dict): Loaded configs from .json file
        _attrs (list of str): A list of the configurations for printing

    """
    dim = 3
    norm = {'name': 'instance', 'affine': True}
    activ = {'name': 'relu'}
    dropout_rate = 0.2
    max_channels = 512
    upsample = {'name': 'nearest', 'align_corners': False}
    pool = {'name': 'max'}
    global_pool = 'avg'
    decimals = 4
    metrics = list()
    loss = 'dice'
    use_gpu = True
    encode_output = True
    eps = 1e-8
    eval_separate = False
    weight_init = {'name': 'kaiming_normal', 'mode': 'fan_in'}
    save_epoch_0 = False
