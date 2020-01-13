# -*- coding: utf-8 -*-

import numpy as np
import warnings


class Buffer:
    """This class implements a list with a fixed length.

    :class:`Buffer` empties the list if the fixed length is reached, i.e., it
    only keeps track of the values from index 0 to the current index within this
    list. Users can only append the list by calling :meth:`append`; if full, it
    will overwrite from the start.

    This class also supports appending arrays as the element.

    Attributes:
        max_length (int): The maximum length of the buffer.

    """
    def __init__(self, max_length):
        self.max_length = max_length
        self._buffer = [None] * self.max_length
        self._ind = -1

    def __len__(self):
        return self._ind + 1

    @property
    def current(self):
        """Returns current value.

        Returns:
            numpy.ndarray: The value at current location.

        """
        if self._ind == -1:
            message = 'Buffer is empty. Return "nan" as the current value.'
            warnings.warn(message, RuntimeWarning)
            return np.nan
        else:
            return np.array(self._buffer[self._ind])

    @property
    def mean(self):
        """Returns the average aross the accumulated values.

        Returns:
            numpy.ndarray: The mean across the accumulated values.

        """
        if self._ind == -1:
            message = 'Buffer is empty. Return numpy.array([]) as the mean.'
            warnings.warn(message, RuntimeWarning)
            return np.array(list())
        else:
            return np.mean(self._buffer[:self._ind+1], axis=0)

    @property
    def all(self):
        """Returns all values.

        Returns:
            numpy.ndarray: All tracked values.

        """
        if self._ind == -1:
            message = 'Buffer is empty. Return numpy.array([]) for all values.'
            warnings.warn(message, RuntimeWarning)
            return np.array(list())
        else:
            return np.stack(self._buffer[:self._ind+1], axis=0)

    def append(self, value):
        """Appends a new element. Overwrites from the start if full.

        Args:
            value (numpy.ndarray or number): The value to append.

        Raises:
            ValueError: The value to append has different shape from previously
                added values.

        """
        if self._ind == self.max_length - 1:
            self._ind = 0
        else:
            self._ind += 1
        value = np.array(value).squeeze()
        if self._ind > 0:
            if value.shape != self._buffer[self._ind - 1].shape:
                m = ('The value to append has different shape from the '
                     'previously added values.')
                raise ValueError(m)
        self._buffer[self._ind] = value
