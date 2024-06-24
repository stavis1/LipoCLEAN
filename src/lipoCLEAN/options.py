#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:44:52 2024

@author: 4vt
"""
import os
import sys
import shutil
from shutil import copy2
import logging

class InputError(Exception):
    pass

class options:
    def __init__(self):
        from argparse import ArgumentParser
        import tomllib
        
        parser = ArgumentParser(prog = 'lipoCLEAN',
                                description = 'A post processing filter for MS-DIAL lipid identifications.',
                                usage = self.help_usage())
        parser.add_argument('-o', '--options', action = 'store', required = False, default = False,
                            help = 'path to options file', metavar='options.toml')
        parser.add_argument('-p', '--print', action = 'store', required = False, default = False,
                            help = 'print a default options file with the specified name and exit', metavar = 'options.toml')
        args = parser.parse_args()
        
        if args.print:
            self.print_options(args.print)
            sys.exit(0)
        elif not args.options:
            raise InputError('One of "--options" or "--print" must be used.')
        
        self.optfile = os.path.abspath(args.options)

        with open(args.options,'rb') as toml:
            options = tomllib.load(toml)
        self.__dict__.update(options)
        
        if hasattr(self.min_rt, '__iter__'):
            self.min_rt = {f:r for f,r in zip(self.data, self.min_rt)}
        else:
            self.min_rt = {f:self.min_rt for f in self.data}
        
        #set up logger
        logging.captureWarnings(True)
        self.logs = logging.getLogger('lipoCLEAN')
        self.logs.setLevel(10)
        formatter = formatter = logging.Formatter('%(asctime)s | %(levelname)s: %(message)s')

        logfile = logging.FileHandler(os.path.join(self.working_directory, 'lipoCLEAN.log'))
        logfile.setLevel(10)
        logfile.setFormatter(formatter)
        self.logs.addHandler(logfile)
        
        logstream = logging.StreamHandler()
        logstream.setLevel(self.log_level)
        logstream.setFormatter(formatter)
        self.logs.addHandler(logstream)
    
    def help_usage(self):
        if getattr(sys, "frozen", False) and hasattr(sys, '_MEIPASS'):
            if os.name == 'nt':
                call_str = 'lipoCLEAN.exe'
            else:
                call_str = 'lipoCLEAN'
        else:
            call_str = 'python -m lipoCLEAN'
        return f'{call_str} [-h] [-o options.toml] [-p options.toml]'
    
    def print_options(self, path):
        if getattr(sys, "frozen", False) and hasattr(sys, '_MEIPASS'):
            resolved_path = os.path.abspath(sys._MEIPASS)
        else:
            resolved_path = os.path.abspath(os.path.dirname(__file__))
        if os.environ.get('AM_I_IN_A_DOCKER_CONTAINER', False):
            example_options = os.path.join(resolved_path, 'docker_example_options.toml')
        else:
            example_options = os.path.join(resolved_path, 'example_options.toml')
        shutil.copy2(example_options, path)

def validate_inputs(args):
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
                'output',
                'QC_plots',
                'QC_plot_extensions',
                'log_level',
                'cutoffs',
                'features']
    problems = [r for r in required if not r in args.__dict__.keys()]
    if problems:
        args.logs.error('Required settings not found in options file:\n' + '\n'.join(problems))
        raise InputError()

    args.logs.debug(f'Started run in mode {args.mode}')
    
    model = os.path.join(args.working_directory, args.model)
    if args.mode == 'infer' and not os.path.exists(model):
        args.logs.error(f'Attempting inference without a pre-trained model, expected {os.path.abspath(model)}.')
        raise InputError()
    
    #prevent overwriting files
    if not args.overwrite:
        if args.mode == 'train' and os.path.exists(model):
            args.logs.error('Overwrite is false and a model with this name already exists')
            raise InputError()
        problems = []
        for file in ['not_considered.tsv', 
                     'positive_lipids.tsv', 
                     'negative_lipids.tsv', 
                     'reanalyze_lipids.tsv',
                     os.path.basename(args.optfile),
                     'QC']:
            path = os.path.abspath(os.path.join(args.output, file))
            if os.path.exists(path):
                problems.append(path)
        if problems:
            args.logs.error('Overwrite is false and these files exist:\n' + '\n'.join(problems))
            raise InputError()

def setup_workspace(args):
    os.chdir(args.working_directory)
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    else:
        args.logs.warning('Preexisting output directory found, files will be overwritten.')
    target_optfile = os.path.join(args.output, os.path.basename(args.optfile))
    if os.path.abspath(target_optfile) != args.optfile:
        copy2(args.optfile, target_optfile)
    
    args.model = os.path.abspath(args.model)
    if args.mode == 'train' and not os.path.exists(args.model):
        os.mkdir(args.model)
    
    if args.QC_plots != 'none':
        qc_path = os.path.join(args.output, 'QC')
        if not os.path.exists(qc_path):
            os.mkdir(qc_path)
        else:
            args.logs.warning('Preexisting QC path found, files will be overwritten.')
        scores_path = os.path.join(qc_path, 'scores_plots')
        if not os.path.exists(scores_path):
            os.mkdir(scores_path)
        if args.QC_plots == 'all':
            perfile_path = os.path.join(qc_path, 'per_file_plots')
            if not os.path.exists(perfile_path):
                os.mkdir(perfile_path)

        
