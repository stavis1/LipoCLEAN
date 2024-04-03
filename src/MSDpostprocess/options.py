#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:44:52 2024

@author: 4vt
"""

from argparse import ArgumentParser
import tomllib
import os
import shelve

class options:
    def __init__(self):
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
                    'overwrite']
        problems = [r for r in required if not r in self.__dict__.keys()]
        if problems:
            raise Exception('Required settings not found in options file:\n' + \
                            '\n'.join(problems))

        os.chdir(self.working_directory)
        if not 'cores' in self.__dict__.keys():
            self.cores = 1
        
        if os.path.exists(self.model) and not self.overwrite:
            raise Exception('A persistant model already exists and your settings do not allow overwrites.')
        
        with shelve.open(self.model, writeback = True) as db:
            db['options'] = self

    @classmethod
    def parse(cls, argstr = ''):
        parser = ArgumentParser()
        parser.add_argument('-o', '--options', action = 'store', required = True,
                            help = 'Path to options file.')
        if argstr:
            args = parser.parse_args(argstr.split())
        else:
            args = parser.parse_args()
        
        with open(args.options,'rb') as toml:
            options = tomllib.load(toml)
        
        os.chdir(options["working_directory"])
        if os.path.exists(options['model']):
            with shelve.open(options['model']) as db:
                return db['options']
        else:
            return cls()

