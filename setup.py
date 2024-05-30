#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 13:14:54 2024

@author: 4vt
"""

import setuptools

setuptools.setup(name="MSDpostprocess",
                 version="0.0.0",
                 url="https://github.com/stavis1/MSDpostprocess",
                 author="Steven Tavis",
                 author_email="stavis@vols.utk.edu",
                 package_dir={"": "src"},
                 packages=setuptools.find_namespace_packages(where="src"),
                 include_package_data=True,
                 license_files=['LICENSE'],
                 classifiers=['License :: OSI Approved :: MIT License',
                              'Operating System :: OS Independent',
                              'Programming Language :: Python :: 3.11'],
                 python_requires='==3.11.9',
                 install_requires=['matplotlib==3.8.4',
                                   'numpy==1.24.4',
                                   'pandas==2.2.2',
                                   'pip==24.0',
                                   'scikit-learn==1.4.2',
                                   'statsmodels==0.14.1',
                                   'brain-isotopic-distribution==1.5.16',
                                   'linear-tree==0.3.5'])



