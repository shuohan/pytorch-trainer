from singleton_config import Config as Config_


class Config(Config_):
    """Global configurations.

    Attributes:
        decimals (int): The number of decimals to print and log.
        use_cuda (bool): Use CUDA GPU to train or validate.
        save_epoch0 (bool): Save before updating any weights.
        num_epochs (int): The number of epochs to train the networks.
        valid_step (int): Validate every this number of epochs.
        save_ckpt_step (int): Save a checkpoint every this number of epochs.
        save_train_step (int): Save training results every this # epochs.
        save_valid_step (int): Save validation results every this # epochs.

    """
    def __init__(self):
        super().__init__()
        self.add_config('decimals', 4)
        self.add_config('use_cuda', True)
        self.add_config('save_epoch0', False)
        self.add_config('num_epochs', 100)
        self.add_config('valid_step', 10)
        self.add_config('save_ckpt_step', 10)
        self.add_config('save_train_step', 10)
        self.add_config('save_valid_step', 10)
        self.add_config('log_samples', False)


# from enum import Enum
    # reduction = Reduction.MEAN
    # enum Reduction: The loss reduction method.

    # dump = True
    # bool: Dump the intermediate results into cpu.
# class Reduction(Enum):
#     MEAN = 'mean'
#     SUM = 'sum'
# 
