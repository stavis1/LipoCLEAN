#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:13:08 2024

@author: 4vt
"""
import sys
from shutil import rmtree
import os
import unittest
import logging

from MSDpostprocess.options import options, setup_workspace
from MSDpostprocess.utilities import read_files


class baseTestSuite(unittest.TestCase):
    def setUp(self):
        self.init_dir = os.getcwd()
        self.init_argv = sys.argv
        test_path = os.path.dirname(__name__)
        sys.argv = [sys.argv[0], '--options', os.path.join(test_path, 'options.toml')]
        self.args = options()
        logger = logging.getLogger('MSDpostprocess')
        logger.setLevel(logging.FATAL)
        os.chdir(self.args.working_directory)
        pass
    
    def tearDown(self):
        os.chdir(self.init_dir)
        sys.argv = self.init_argv
        pass

class workspaceTestSuite(baseTestSuite):
    def setUp(self):
        super().setUp()
        self.lipid_data = read_files(self.args)
        setup_workspace(self.args)
    
    def tearDown(self):
        rmtree(self.args.output)
        os.remove('MSDpostprocess.log')
        super().tearDown()
