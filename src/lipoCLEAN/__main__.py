#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:14:41 2024

@author: 4vt
"""
from lipoCLEAN.options import options, setup_workspace, validate_inputs
args = options()

from lipoCLEAN.utilities import read_files, filter_data, split_index, write_data
from lipoCLEAN.models import mz_correction, rt_correction, predictor_model, add_isotope_error
from lipoCLEAN.QC import plot_mz_QC, plot_rt_QC, plot_final_QC, plot_pairwise_scores

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
    lipid_data['split'] = ['train' if i in train_idx else 'test' for i in lipid_data.index]
    
    mz_model.fit(lipid_data.loc[train_idx])
    mz_model.dump()
    mz_model.assess(lipid_data.loc[train_idx], 'Training')
    if args.test_split < 1:
        mz_model.assess(lipid_data.loc[test_idx], 'Test')
    lipid_data = mz_model.correct_data(lipid_data)

    rt_model.fit(lipid_data.loc[train_idx])
    rt_model.dump()
    rt_model.assess(lipid_data.loc[train_idx], 'Training')
    if args.test_split < 1:
        rt_model.assess(lipid_data.loc[test_idx], 'Test')
    lipid_data = rt_model.correct_data(lipid_data)
    
    final_model.fit(lipid_data.loc[train_idx])
    final_model.dump()
    final_model.assess(lipid_data.loc[train_idx], 'Training')
    if args.test_split < 1:
        final_model.assess(lipid_data.loc[test_idx], 'Test')
    
else:
    mz_model.load()
    rt_model.load()
    final_model.load()
    lipid_data = mz_model.correct_data(lipid_data)
    lipid_data = rt_model.correct_data(lipid_data)

lipid_data = final_model.classify(lipid_data)
write_data(lipid_data, args)

if args.QC_plots != 'none':
    plot_mz_QC(mz_model, args)
    plot_rt_QC(rt_model, args)
    plot_final_QC(final_model, args)
    plot_pairwise_scores(lipid_data, args)