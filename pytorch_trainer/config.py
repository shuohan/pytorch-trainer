# -*- coding: utf-8 -*-

from config import Config as Config_


class Config(Config_):
    """Global configurations.

    >>> from pytorch_trainer import Config
    >>> Config.attribute = new_value

    to update the configurations.

    """
    decimals = 4
    """int: The number of decimals to print and log."""
    use_gpu = True
    """bool: Use GPU to train and validate."""
    save_epoch_0 = False
    """bool: Save before updating any weights."""
