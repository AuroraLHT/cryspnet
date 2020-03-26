import os

from setuptools import setup
from setuptools import find_packages


with open('README.md') as f:
    readme = f.read()

with open('requirement.txt') as f:
    requirements = f.readlines()

setup(
    name='cryspnet',
    version='0.1',
    description='CRYSPNet: Crystal Structure Predictions via Neural Network. Liang et al. (2020).',
    long_description=readme,
    author='H Liang, V Stanev, A. G Kusne, and I Takeuchi',
    author_email='auroralht@gmail.com, vstanev@umd.edu, aaron.kusne@nist.gov, takeuchi@umd.edu',
    packages=find_packages(),
    install_requires=[
          requirements,
    ],
    include_package_data=True,

    classifiers=['Programming Language :: Python :: 3.6',
                 'Development Status :: 4 - Beta',
                 'Intended Audience :: Science/Research',
                 'Intended Audience :: System Administrators',
                 'Intended Audience :: Information Technology',
                 'Operating System :: OS Independent',
                 'Topic :: Other/Nonlisted Topic',
                 'Topic :: Scientific/Engineering'],
    )
