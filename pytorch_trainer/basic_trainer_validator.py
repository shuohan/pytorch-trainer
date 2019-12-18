# -*- coding: utf-8 -*-

from .config import Config
from .trainer_validator import Trainer, Validator


class _BasicInterface:
    """An interface class.

    This class is for :class:`BasicTrainer` and :class:`BasicValidator`.

    """
    def _dump_results(self, input, output, truth):
        """Dumps intermediate results at mini-batch."""
        if Config.dump:
            self._dump('input', input)
            self._dump('output', output)
            self._dump('truth', truth)


class BasicTrainer(Trainer, _BasicInterface):
    """A basic trainer.

    This trainer has only one model to train. Correspondingly, it also only
    accepts only one loss function to update this model. The :attr:`data_loader`
    should yield a :class:`tuple` of input and truth tensors.

    Notes:
        Multiple loss terms should be defined within :attr:`loss_func`.

    Attributes:
        loss_func (function): The loss function.
        optim (torch.optim.Optimizer): The optimizer.

    """
    def __init__(self, model, loss_func, optim, data_loader, num_epochs=500):
        super().__init__(data_loader, num_epochs)
        self.models['model'] = model
        self.losses['model'] = Buffer(self.num_batches)
        self.loss_func = loss_func
        self.optim = optim

    def _train_on_batch(self, data):
        """Trains the model for each batch.
        
        Args:
            data (tuple[torch.Tensor]): The first element is the input tensor to
                the model. The second element is the truth output of the model.

        """
        input, truth = data[0], data[1]
        output = self.models['model'](input)
        loss = self.loss_func(output, truth)
        self.optim.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.losses['model'].append(self._numpy(loss))
        self._dump_results(input, output, truth)


class BasicValidator(Validator, _BasicInterface):
    """A basic validator.

    This validator has only one model to validate with an input and a truth.
    This class is used with :class:`BasicTrainer`.

    """
    def _validate(self, data):
        """Validates the model.

        Args:
            data (tuple[torch.Tensor]): The first element is the input tensor to
                the model. The second element is the truth output of the model.

        """
        input, truth = data[0], data[1]
        output = self.observable.models['model'](input)
        loss = self.observable.loss_func(output, truth)
        self.losses['model'].append(self._numpy(loss))
        self._dump_results(input, output, truth)
