#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 12:26:02 2024

@author: 4vt
"""
import numpy as np
import pandas as pd
import os

def read_files(args):
    dfs = []
    for i,file in enumerate(args.data):
        #read data and accout for potential metadata headers
        with open(file, 'r') as text:
            line = text.readline().split('\t')
            if line[0]:
                lipid_data = pd.read_csv(file, sep = '\t')
            else:
                lipid_data = pd.read_csv(file, sep = '\t', skiprows = 4)
        
        #find and rename m/z matrix columns
        mz_cols = list(lipid_data.columns)[list(lipid_data.columns).index('MS/MS spectrum')+1:]
        if any(lipid_data[c].dtype.kind != 'f' for c in mz_cols):
            print('Invalid data types found in m/z columns. This could be due to non m/z columns being inserted after "MS/MS spectrum". Please check', flush = True)
            print('m/z columns:\n' + '\n'.join(mz_cols))
            raise Exception()
    
        
        newcols = list(lipid_data.columns)[:list(lipid_data.columns).index('MS/MS spectrum')+1]
        newcols.extend([f'observed_mz_F{i}_{c}' for c in mz_cols])
        lipid_data.columns = newcols
        lipid_data['file'] = [file]*lipid_data.shape[0]
        dfs.append(lipid_data)
    lipid_data = pd.concat(dfs, ignore_index = True)
    return lipid_data

def filter_data(lipid_data, args):
    #first pass filter for extremely low confidence IDs
    bad_idx = []
    #must have MS2 data for ID, matches based on RT and exact mass are not considered valid
    bad_idx.extend(lipid_data[[not(mz and rt) for mz,rt in zip(lipid_data['m/z matched'],lipid_data['RT matched'])]].index)
    #MS-Dial annotates some features that don't have a formula, we do not consider this an ID
    bad_idx.extend(lipid_data[[(('D' in f) or ('i' in f)) if type(f) == str else True for f in lipid_data['Formula']]].index) #this breaks brainpy
    #features with an unknown ontology are also not considered IDs
    bad_idx.extend(lipid_data[[type(o) != str or o in ['Unknown','Others'] for o in lipid_data['Ontology']]].index)
    #features eluting in the dead volume are not considered reliably identifiable in our data and are discarded
    #this is an optional runtime argument if unused min_rt = 0 and this line does nothing
    bad_idx.extend(lipid_data[[r < args.min_rt for r in lipid_data['Average Rt(min)']]].index)
    #these columns need to have valid numbers in them for the model to work
    nonnancols = ['Dot product', 'S/N average', 'Average Rt(min)', 'Reference m/z']
    bad_idx.extend(lipid_data[[any(np.isnan(v)) for v in zip(*[lipid_data[c] for c in nonnancols])]].index)
    bad_idx.extend(lipid_data[[type(s) != str for s in lipid_data['MS1 isotopic spectrum']]].index)
    bad_idx = list(set(bad_idx))
    print(f'{len(bad_idx)} entries not considered\n', flush = True)

    bad_rows = lipid_data.loc[bad_idx]
    bad_rows.to_csv(os.path.join(args.output, 'not_considered.tsv'), sep = '\t', index = False)
    lipid_data.drop(bad_idx, inplace=True)
    return lipid_data

def split_index(lipid_data, args):
    rng = np.random.default_rng(args.seed)

    train_idx = rng.choice(lipid_data.index, 
                           int(lipid_data.shape[0]*args.test_split),
                           replace = False)
    test_idx = set(lipid_data.index).difference(train_idx)
    return (np.array(train_idx), np.array(list(test_idx)))

def write_data(lipid_data, args):
    #move newly created columns to the beginning of the table
    last_mz_col = [c for c in lipid_data.columns if c.startswith('observed_mz_')][-1]
    last_mz_idx = list(lipid_data.columns).index(last_mz_col) + 1
    new_cols = list(lipid_data.columns[last_mz_idx:])
    old_cols = list(lipid_data.columns[:last_mz_idx])
    lipid_data = lipid_data[new_cols + old_cols]
    
    #split data by classification and write files
    groups = lipid_data.groupby('class')
    categories = {-1:'reanalyze_lipids.tsv',
                  0:'negative_lipids.tsv',
                  1:'positive_lipids.tsv'}
    for cat, data in groups:
        data.to_csv(os.path.join(args.output, categories[cat]), sep = '\t', index = False)

