#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:14:41 2024

@author: 4vt
"""
import sys
sys.argv.extend('--options /home/4vt/Documents/data/SLT05_MSDpostprocess/MSDpostprocess/example_data/options.toml'.split())

from MSDpostprocess.options import options, setup_workspace, validate_inputs
from MSDpostprocess.utilities import read_files, filter_data, split_index, write_data
from MSDpostprocess.models import mz_correction, rt_correction, predictor_model, add_isotope_error
from MSDpostprocess.QC import plot_mz_QC, plot_rt_QC, plot_final_QC, plot_pairwise_scores

args = options()
validate_inputs(args)
setup_workspace(args)

lipid_data = read_files(args)
lipid_data = filter_data(lipid_data, args)
lipid_data = add_isotope_error(lipid_data)

mz_model = mz_correction(args)
rt_model = rt_correction(args)
final_model = predictor_model(args)   

if args.mode == 'train':
    train_idx, test_idx = split_index(lipid_data, args)
    
    mz_model.fit(lipid_data.loc[train_idx])
    mz_model.dump()
    mz_model.assess(lipid_data.loc[train_idx], 'Training')
    mz_model.assess(lipid_data.loc[test_idx], 'Test')
    lipid_data = mz_model.correct_data(lipid_data)

    rt_model.fit(lipid_data.loc[train_idx])
    rt_model.dump()
    rt_model.assess(lipid_data.loc[train_idx], 'Training')
    rt_model.assess(lipid_data.loc[test_idx], 'Test')
    lipid_data = rt_model.correct_data(lipid_data)
    
    final_model.fit(lipid_data.loc[train_idx])
    final_model.dump()
    final_model.assess(lipid_data.loc[train_idx], 'Training')
    final_model.assess(lipid_data.loc[test_idx], 'Test')
    
else:
    mz_model.load()
    rt_model.load()
    final_model.load()
    lipid_data = mz_model.correct_data(lipid_data)
    lipid_data = rt_model.correct_data(lipid_data)

lipid_data = final_model.classify(lipid_data)
write_data(lipid_data, args)

if args.QC_plots:
    plot_mz_QC(mz_model, args)
    plot_rt_QC(rt_model, args)
    plot_final_QC(final_model, args)
    plot_pairwise_scores(lipid_data, args)