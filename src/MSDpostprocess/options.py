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
        
        parser = ArgumentParser()
        parser.add_argument('-o', '--options', action = 'store', required = True,
                            help = 'Path to options file.')
        args = parser.parse_args()
        
        with open(args.options,'rb') as toml:
            options = tomllib.load(toml)
        self.__dict__.update(options)
        
        required = ['working_directory',
                    'data',
                    'seed',
                    'min_rt',
                    'lowess_frac',
                    'ppm',
                    'cutoff_low',
                    'cutoff_high',
                    'model',
                    'overwrite',
                    'mode',
                    'test_split',
                    'cores']
        problems = [r for r in required if not r in self.__dict__.keys()]
        if problems:
            raise Exception('Required settings not found in options file:\n' + \
                            '\n'.join(problems))

        os.chdir(self.working_directory)
        
        if self.mode == 'inference' and not os.path.exists(self.model):
            raise Exception('Attempting inference without a pre-trained model.')
