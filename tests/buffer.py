#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pytorch_engine.training.buffer import Buffer

buffer = Buffer(3)
print('empty')
print('length', len(buffer), 'current', buffer.current, 'mean', buffer.mean)

buffer.append(1.1)
print('1')
print('length', len(buffer), 'current', buffer.current, 'mean', buffer.mean)

buffer.append(2.2)
print('2')
print('length', len(buffer), 'current', buffer.current, 'mean', buffer.mean)

buffer.append(3.3)
print('3')
print('length', len(buffer), 'current', buffer.current, 'mean', buffer.mean)

buffer.append(4.4)
print('4')
print('length', len(buffer), 'current', buffer.current, 'mean', buffer.mean)
