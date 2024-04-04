#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:14:41 2024

@author: 4vt
"""
import sys
sys.argv.extend('--options /home/4vt/Documents/data/SLT05_MSDpostprocess/MSDpostprocess/example_data/options.toml'.split())

from MSDpostprocess.options import options
args = options()

from MSDpostprocess import utilities
from MSDpostprocess.models import mz_correction, rt_correction, predictor_model

lipid_data = utilities.read_files(args)
lipid_data = utilities.filter_data(lipid_data, args)

if args.mode == 'train':
    train_data, test_data = utilities.split_data(lipid_data, args)
    
