#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: David Corre, IAP, corre@iap.fr

The setup script.
"""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
        'numpy',
        'astropy',
        'matplotlib',
        'Sphinx',
        'twine',
        'h5py',
        'keras',
        'tensorflow',
        'opencv-python-headless',
        'multidict',
        'async-timeout',
        'attrs',
        'chardet',
       ]
setup_requirements = [
        'pytest-runner',
        'flake8',
        'bumpversion',
        'wheel',
        'twine']

test_requirements = [
        'pytest',
        'pytest-cov',
        'pytest-console-scripts',
        'pytest-html',
        'watchdog']

setup(
    author="David Corre",
    author_email='david.corre.fr@gmail.com',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="CNN based tools to help identification of astronomical transients.",
    entry_points={
        'console_scripts': [
            'tbd_cnn-convert = tbd_cnn.cli.convert:main',
            'tbd_cnn-train = tbd_cnn.cli.train:main',
            'tbd_cnn-infer = tbd_cnn.cli.infer:main',
            'tbd_cnn-checkinfer = tbd_cnn.cli.checkinfer:main',
            'tbd_cnn-diagonstic = tbd_cnn.cli.diagnostic:main',
            'tbd_cnn-optimize_size = tbd_cnn.cli.optimize_size:main',
            'tbd_cnn-plot_results = tbd_cnn.cli.plot_results:main',

        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords=['transients', 'detection pipeline', 'astronomy',
              'CNN'],
    name='tbd_cnn',
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    #url='https://github.com/dcorre/gmadet',
    version='0.1.0',
    zip_safe=False,
)
