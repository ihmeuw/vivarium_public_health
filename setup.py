#!/usr/bin/env python

from setuptools import setup, find_packages


setup(name='ceam_public_health',
      version='0.1',
      packages=find_packages(),
      include_package_data=True,
      install_requires=[
          'pandas',
          'numpy',
          'scipy',
          'tables'
      ],
      )

