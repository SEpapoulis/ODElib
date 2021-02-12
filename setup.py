#!/usr/bin/env python

import os
from setuptools import setup

def read(fname):
    try:
        with open(os.path.join(os.path.dirname(__file__), fname)) as fh:
            return fh.read()
    except IOError:
        return ''

requirements = read('requirements.txt').splitlines()

setup(name='ODElib',
      version='0.1',
      description='Python Distribution Utilities',
      author='Spiridon Papoulis',
      author_email='spapouli@vols.utk.edu',
      url='https://github.com/SEpapoulis/ODElib',
      packages=['ODElib'],
      license='GPL-3.0 License',
      install_requires=requirements,
      python_requires='>=3.7'
     )
