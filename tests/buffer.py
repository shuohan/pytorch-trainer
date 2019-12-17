#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pytorch_trainer.buffer import Buffer


def test_buffer_scalar():
    buffer = Buffer(3)
    assert len(buffer) == 0
    assert np.isnan(buffer.current)
    assert buffer.mean.size == 0
    assert buffer.all.size == 0

    buffer.append(0.1)
    assert len(buffer) == 1
    assert buffer.current == 0.1
    assert buffer.mean == 0.1
    assert buffer.all == [0.1]

    buffer.append(0.2)
    assert len(buffer) == 2
    assert buffer.current == 0.2
    assert np.isclose(buffer.mean, 0.15)
    assert buffer.all == [0.1, 0.2]

    buffer.append(0.3)
    assert len(buffer) == 3
    assert buffer.current == 0.3
    assert np.isclose(buffer.mean, 0.20)
    assert buffer.all == [0.1, 0.2, 0.3]

    buffer.append(0.4)
    assert len(buffer) == 1
    assert buffer.current == 0.4
    assert buffer.mean == 0.4
    assert buffer.all == [0.4]


def test_buffer_array():
    buffer = Buffer(3)
    assert len(buffer) == 0
    assert np.isnan(buffer.current)
    assert buffer.mean.size == 0
    assert buffer.all.size == 0

    buffer.append(np.array([[1, 2, 3], [4, 5, 6]]))
    assert len(buffer) == 1
    assert (buffer.current == np.array([[1, 2, 3], [4, 5, 6]])).all()
    assert (buffer.mean == np.array([[1, 2, 3], [4, 5, 6]])).all()
    all = np.vstack(buffer.all)
    assert (all == np.vstack(np.array([[1, 2, 3], [4, 5, 6]]))).all()

    buffer.append(np.array([[11, 12, 13], [14, 15, 16]]))
    assert len(buffer) == 2
    assert (buffer.current == np.array([[11, 12, 13], [14, 15, 16]])).all()
    assert np.isclose(buffer.mean, np.array([[6, 7, 8], [9, 10, 11]])).all()
    all = np.stack(buffer.all, axis=0)
    ref = np.stack([np.array([[1, 2, 3], [4, 5, 6]]),
                    np.array([[11, 12, 13], [14, 15, 16]])], axis=0)
    assert (all == ref).all()

    buffer.append(np.array([[21, 22, 23], [24, 25, 26]]))
    assert len(buffer) == 3
    assert (buffer.current == np.array([[21, 22, 23], [24, 25, 26]])).all()
    ref = np.array([[11, 12, 13], [14, 15, 16]])
    assert np.isclose(buffer.mean, ref).all()
    all = np.stack(buffer.all, axis=0)
    ref = np.stack([np.array([[1, 2, 3], [4, 5, 6]]),
                    np.array([[11, 12, 13], [14, 15, 16]]),
                    np.array([[21, 22, 23], [24, 25, 26]])], axis=0)
    assert (all == ref).all()

    buffer.append(np.array([[31, 32, 33], [34, 35, 36]]))
    assert len(buffer) == 1
    assert (buffer.current == np.array([[31, 32, 33], [34, 35, 36]])).all()
    assert (buffer.mean == np.array([[31, 32, 33], [34, 35, 36]])).all()
    all = np.vstack(buffer.all)
    assert (all == np.vstack(np.array([[31, 32, 33], [34, 35, 36]]))).all()


def test_wrong_shape():
    buffer = Buffer(3)
    buffer.append(1)
    try:
        buffer.append(np.array([1, 2, 3]))
    except ValueError:
        print('Wrong shape')
        assert True


if __name__ == '__main__':
    test_buffer_scalar()
    test_buffer_array()
    test_wrong_shape()
