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
import re

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

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
        os.remove('MSDpostprocess.log')
        pass

class hasWorkspaceTestSuite(baseTestSuite):
    def setUp(self):
        super().setUp()
        self.lipid_data = read_files(self.args)
        setup_workspace(self.args)
    
    def tearDown(self):
        rmtree(self.args.output)
        rmtree(self.args.model)
        super().tearDown()

class modelTestSuite(hasWorkspaceTestSuite):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def setUp(self):
        super().setUp()
        self.model = self.model_class(self.args)
    
    def get_features(self):
        init_argv = sys.argv
        test_path = os.path.dirname(__name__)
        sys.argv = [sys.argv[0], '--options', os.path.join(test_path, 'options.toml')]
        args = options()
        sys.argv = init_argv
        classname = str(self.model_class(args).__class__)
        self.model_name = re.search(r"\.([^.]+)'>\Z", classname).group(1)
        self.features = args.features[self.model_name]
    
    def make_fake_data(self, m1, m2, rng):
        N = 100
        pos = rng.normal(m1, 1, (N, len(self.features)))
        neg = rng.normal(m2, 1, (N, len(self.features)))
        labels = np.array([1]*N + [0]*N)
        idx = rng.choice(range(N*2), size = N*2, replace = False)
        features = np.concatenate((pos,neg))
        features = features[idx]
        labels = labels[idx]
        data = pd.DataFrame(features, columns = self.features)
        data['label'] = labels
        return data
    
    def fit_on_data(self, m1, m2):
        rng = np.random.default_rng(1)

        data = self.make_fake_data(m1, m2, rng)
        self.model.fit(data)
        
        data = self.make_fake_data(m1, m2, rng)
        scores = self.model._predict_prob(data)
        aucroc = roc_auc_score(data['label'], scores)
        
        calls = self.model.predict(data)
        n_incorrect = np.sum(calls != data['label'].to_numpy())
        
        return (aucroc, n_incorrect)
    
    def test_fit_on_separable_data(self):
        aucroc, n_incorrect = self.fit_on_data(0, 4)
        
        with self.subTest(msg = 'Testing AUC-ROC is good'):
            self.assertAlmostEqual(aucroc, 1, delta = 0.02)
        
        with self.subTest(msg = 'Testing N incorrect is good'):
            self.assertTrue(n_incorrect < 10)

    def test_fit_on_inseparable_data(self):
        aucroc, n_incorrect = self.fit_on_data(0, 0)
        
        with self.subTest(msg = 'Testing AUC-ROC ~ 0.5'):
            self.assertAlmostEqual(aucroc, 0.5, delta = 0.1)
        
        with self.subTest(msg = 'Testing that N incorrect is high'):
            self.assertTrue(n_incorrect > 80)
    
    def test_save_model(self):
        rng = np.random.default_rng(1)
        with self.subTest(msg = 'Testing the absence of a preexisting model'):
            files = os.listdir(self.args.model)
            self.assertEqual(files, [])
        
        init_aucroc, init_n_incorrect = self.fit_on_data(0, 4)
        
        self.model.dump()        
        self.model = None
        self.model = self.model_class(self.args)
        self.model.load()
        
        data = self.make_fake_data(0, 4, rng)
        scores = self.model._predict_prob(data)
        reload_aucroc = roc_auc_score(data['label'], scores)
        
        calls = self.model.predict(data)
        reload_n_incorrect = np.sum(calls != data['label'].to_numpy())
        
        with self.subTest(msg = 'Testing AUC-ROC equality'):
            self.assertAlmostEqual(init_aucroc, reload_aucroc, delta = 0.05)
        with self.subTest(msg = 'Testing N incorrect equality'):
            self.assertAlmostEqual(init_n_incorrect, reload_n_incorrect, delta = 3)


