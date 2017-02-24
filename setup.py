#!/usr/bin/env python
# ~/ceam/setup.py

from setuptools import setup, find_packages


setup(name='ceam_public_health',
        version='0.1',
        packages=find_packages(),
        include_package_data=True,
        package_index='http://dev-tomflem.ihme.washington.edu/simple/',
        install_requires=[
            'pandas',
            'numpy',
            'scipy',
        ],
        dependency_links=[
            'ssh://git@stash.ihme.washington.edu:7999/cste/ceam-inputs.git#egg=ceam_inputs',
        ]
     )


# End.
