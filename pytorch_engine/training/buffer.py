# -*- coding: utf-8 -*-

import numpy as np
from collections import OrderedDict

from ..config import Config


class Buffer:
    """Array with fixed length

    Call `append` to append a number; if full, it will overwrite from the start.

    Attributes:
        decimals (int): The precision of returned float number
        max_length (int): The maximum length of the buffer
        _buffer (np.array): The internal buffer holding the numbers
        _ind (int): The position index of self._buffer

    """
    def __init__(self, max_length):
        self.decimals = Config().decimals
        self.max_length = max_length
        self._buffer = [None] * self.max_length
        self._ind = -1

    def __len__(self):
        return self._ind + 1

    @property
    def current(self):
        """Return the currect value"""
        if self._ind == -1:
            print(self.__class__, "is empty")
            return float('nan')
        else:
            return np.round(self._buffer[self._ind], self.decimals)

    @property
    def mean(self):
        """Return the average accumulated value"""
        if self._ind == -1:
            print(self.__class__, "is empty")
            return float('nan')
        else:
            return np.round(np.mean(self._buffer[:self._ind+1]), self.decimals)

    @property
    def all(self):
        all = np.round(self._buffer[:self._ind+1], self.decimals)
        all = np.vstack(all)
        return all

    def append(self, current):
        """Add the current value"""
        if self._ind == self.max_length - 1:
            self._ind = 0
        else:
            self._ind += 1
        self._buffer[self._ind] = current
