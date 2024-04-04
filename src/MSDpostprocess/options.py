#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:44:52 2024

@author: 4vt
"""

class options:
    def __init__(self):
        from argparse import ArgumentParser
        import tomllib
        import os
        from shutil import copy2
        
        parser = ArgumentParser()
        parser.add_argument('-o', '--options', action = 'store', required = True,
                            help = 'Path to options file.')
        args = parser.parse_args()
        
        with open(args.options,'rb') as toml:
            options = tomllib.load(toml)
        self.__dict__.update(options)
        
        #check that the options toml is valid
        required = ['working_directory',
                    'data',
                    'seed',
                    'min_rt',
                    'lowess_frac',
                    'ppm',
                    'model',
                    'overwrite',
                    'mode',
                    'test_split',
                    # 'cores',
                    'output']
        problems = [r for r in required if not r in self.__dict__.keys()]
        if problems:
            raise Exception('Required settings not found in options file:\n' + \
                            '\n'.join(problems))

        optfile = os.path.abspath(args.options)
        os.chdir(self.working_directory)
        
        if self.mode == 'infer' and not os.path.exists(self.model):
            raise Exception('Attempting inference without a pre-trained model.')
        
        #prevent overwriting files
        if not self.overwrite:
            if self.mode == 'train' and os.path.exists(self.model):
                raise Exception('Overwrite is false and a model with this name already exists')
            problems = []
            for file in ['not_considered.tsv', 
                         'positive_lipids.tsv', 
                         'negative_lipids.tsv', 
                         'reanalyze_lipids.tsv',
                         os.path.basename(args.options)]:
                path = os.path.abspath(os.path.join(self.output, file))
                if os.path.exists(path):
                    problems.append(path)
            if problems:
                raise Exception('Overwrite is false and these files exist:\n' + '\n'.join(problems))
        
        if not os.path.exists(self.output):
            os.mkdir(self.output)
        copy2(optfile, os.path.join(self.output, os.path.basename(args.options)))
        