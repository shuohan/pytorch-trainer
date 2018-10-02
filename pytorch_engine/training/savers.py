#!/usr/bin/env python
# -*- coding: utf-8 -*-


class ModelSaver:

    def __init__(self, saving_period, saving_path_pattern,
                 save_weights_only=False):
        self.saving_period = saving_period
        self.saving_path_pattern = saving_path_pattern
        self.save_weights_only = save_weights_only

        dirname = os.path.dirname(self.saving_path_pattern)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

    def save(self, epoch, model):
        """Save"""
        if '{epoch}' in self.saving_path_pattern:
            filename = self.saving_path_pattern.format(epoch=epoch)
        else:
            filename = self.saving_path_pattern

        if (epoch + 1) % self.saving_period:
            if self.save_weights_only:
                torch.save(model.state_dict(), filename)
            else:
                torch.save(model, filename)
