# -*- coding: utf-8 -*-

from distutils.core import setup
from glob import glob
import subprocess

scripts = glob('scripts/*')
command = ['git', 'describe', '--tags']
version = subprocess.check_output(command).decode().strip()

setup(name='pytorch-net-engine',
            version=version,
            description='PyTorch convolutional network engine',
            author='Shuo Han',
            author_email='shan50@jhu.edu',
            scripts=scripts,
            install_requires=['torch', 'torchvision'],
            packages=['pytorch_net_engine'])
