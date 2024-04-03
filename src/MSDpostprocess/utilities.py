#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 12:26:02 2024

@author: 4vt
"""

class data:
    def __init__(self, args):
        import pandas as pd
        import numpy as np
        rng = np.random.default_rng(args.seed)
        
        dfs = []
        for file in args.data:
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

            lipid_data['file'] = [file]*lipid_data.shape[0]
            
            newcols = list(lipid_data.columns)[:list(lipid_data.columns).index('MS/MS spectrum')+1]
            newcols.append('file')
            newcols.extend([f'mz_{i}' for i in range(len(mz_cols))])
            lipid_data.columns = newcols
            
            dfs.append(lipid_data)
        
        self.lipid_data = pd.concat(dfs)
        
        if args.mode == 'training':
            split_idx = 

