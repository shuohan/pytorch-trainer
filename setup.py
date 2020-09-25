from setuptools import setup, find_packages
from glob import glob
import subprocess

scripts = glob('scripts/*')
command = ['git', 'describe', '--tags']
version = subprocess.check_output(command).decode().strip()

setup(name='pytorch-trainer',
      version=version,
      description='PyTorch trainer',
      author='Shuo Han',
      author_email='shan50@jhu.edu',
      scripts=scripts,
      install_requires=['torch', 'torchvision'],
      packages=find_packages(include=['pytorch_trainer']))
