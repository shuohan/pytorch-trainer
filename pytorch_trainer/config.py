# -*- coding: utf-8 -*-

from enum import Enum
from config import Config as Config_


class LoggerFormat(Enum):
    LONG = 'long'
    WIDE = 'wide'


class Reduction(Enum):
    MEAN = 'mean'
    SUM = 'sum'


class Config(Config_):
    """Global configurations.

    >>> from pytorch_trainer import Config
    >>> Config.attribute = new_value

    to update the configurations.

    Note:
        Avoid copying the value of a :class:`Config` attribute to another
        variable, since this variable would not be updated automatically when
        :class:`Config` changes.

    """
    decimals = 4
    """int: The number of decimals to print and log."""
    use_gpu = True
    """bool: Use GPU to train and validate."""
    save_epoch_0 = False
    """bool: Save before updating any weights."""
    num_epochs = 100
    """int: The number of epochs to train the networks."""
    dump = True
    """bool: Dump the intermediate results into cpu."""
    logger_fmt = LoggerFormat.WIDE
    """enum LoggerFormat: The format of logger contents."""
    val_period = 10
    """int: The validation periord in number of epochs."""
    model_period = 10
    """int: The model saving periord in number of epochs."""
    pred_period = 10
    """int: The prediction saving periord in number of epochs."""
    reduction = Reduction.MEAN
    """enum Reduction: The loss reduction method."""
