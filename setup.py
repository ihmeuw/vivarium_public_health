from setuptools import setup, find_packages


setup(
    name='vivarium_public_health',
    version='0.5',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'pandas',
        'numpy',
        'scipy',
        'tables'
    ],
)

