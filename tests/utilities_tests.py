#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:36:53 2024

@author: 4vt
"""
import os
import unittest

import pandas as pd

from lipoCLEAN.utilities import filter_data, read_files, split_index, write_data
from tests import baseTestSuite, hasWorkspaceTestSuite

class readTestSuite(baseTestSuite):    
    def test_read_data(self):
        data = read_files(self.args)
        self.assertEqual(data.shape, (598, 146))
    
    def test_read_multiple_files(self):
        self.args.data = [f'../tests/test_input_small_{i}.tsv' for i in range(1,4)]
        data = read_files(self.args)
        self.assertEqual(data.shape, (15, 370))

class filterTestSuite(hasWorkspaceTestSuite):   
    def test_filter(self):
        filtered = filter_data(self.lipid_data.copy(), self.args)
        self.assertEqual(filtered.shape, (543,146))

class splitTestSuite(hasWorkspaceTestSuite):
    def setUp(self):
        super().setUp()
        self.lipid_data = filter_data(self.lipid_data, self.args)
    
    def test_split_data(self):
        train_idx, test_idx = split_index(self.lipid_data, self.args)
        with self.subTest():
            self.assertEqual((len(train_idx), len(test_idx)), (434, 109))
        with self.subTest():
            self.assertEqual(len(set(train_idx).intersection(test_idx)), 0)

class writeTestSuite(hasWorkspaceTestSuite):
    def test_write_data(self):
        classes = [0]*100 + [1]*100 + [-1]*398
        self.lipid_data['class'] = classes
        write_data(self.lipid_data, self.args)
        with self.subTest():
            neg_path = self.args.output + '/negative_lipids.tsv'
            self.assertTrue(os.path.exists(neg_path))
        with self.subTest():
            rean_path = self.args.output + '/reanalyze_lipids.tsv'
            self.assertTrue(os.path.exists(rean_path))
        with self.subTest():
            pos_path = self.args.output + '/positive_lipids.tsv'
            self.assertTrue(os.path.exists(pos_path))
        
        negative = pd.read_csv(neg_path, sep = '\t')
        with self.subTest():
            self.assertEqual(negative.shape, (100,147))
        reanalyze = pd.read_csv(rean_path, sep = '\t')
        with self.subTest():
            self.assertEqual(reanalyze.shape, (398,147))
        positive = pd.read_csv(pos_path, sep = '\t')
        with self.subTest():
            self.assertEqual(positive.shape, (100,147))

    
if __name__ == "__main__":
    unittest.main()
