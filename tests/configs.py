#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pytorch_engine import Config

config = Config('input.json')
assert config.norm == {'name': 'batch'}
print(config)
config.norm['name'] = 'instance'
assert Config().norm == {'name': 'instance'}
config.save('output.json')
config = Config()
config.load('output.json')
print('-' * 80)
print(config)
assert config.norm == {'name': 'instance'}
