#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:26:55 2024

@author: 4vt
"""

import os
import unittest
from shutil import rmtree

from MSDpostprocess.options import validate_inputs, setup_workspace, InputError
from tests import baseTestSuite

class validateTestSuite(baseTestSuite):
    def test_error_on_missing_inputs(self):
        for attr in ['working_directory',
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
                    'features']:
            tmp = self.args.__dict__[attr]
            del self.args.__dict__[attr]
            with self.subTest():
                with self.assertRaises(InputError):
                    validate_inputs(self.args)
            setattr(self.args, attr, tmp)
            
    def test_error_on_missing_model(self):
        self.args.mode = 'infer'
        with self.assertRaises(InputError):
            validate_inputs(self.args)

    def test_error_on_model_overwrite(self):
        try:
            self.args.overwrite = False
            os.mkdir(self.args.model)
            with self.assertRaises(InputError):
                validate_inputs(self.args)
        finally:
            os.rmdir(self.args.model)
    
    def test_error_on_data_overwrite(self):
        try:
            self.args.overwrite = False
            setup_workspace(self.args)
            test_file = os.path.join(self.args.output,'reanalyze_lipids.tsv')
            with open(test_file, 'w') as tsv:
                tsv.write('TEST')
            with self.assertRaises(InputError):
                validate_inputs(self.args)
        finally:
            rmtree(self.args.output)
            rmtree(self.args.model)


if __name__ == "__main__":
    unittest.main()