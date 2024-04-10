#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 09:28:54 2024

@author: 4vt
"""

import unittest

from MSDpostprocess.QC import TPR, FPR, FDR
from tests import baseTestSuite

class metricsTestSuite(baseTestSuite):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputs = [(0,0,0,0),
                       (10,10,10,10),
                       (10,10,0,0),
                       (0,0,10,10),
                       (10,0,10,0),
                       (0,0,0,10),
                       (0,10,0,0)]
    
    def test_TPR(self):
        outputs = [0, 0.5, 0, 0.5, 0, 1, 0]
        for i,o in zip(self.inputs, outputs):
            tn,fp,fn,tp = i
            with self.subTest():
                self.assertEqual(TPR(tn,fp,fn,tp), o)
    
    def test_FPR(self):
        outputs = [0, 0.5, 0.5, 0, 0, 0, 1]
        for i,o in zip(self.inputs, outputs):
            tn,fp,fn,tp = i
            with self.subTest():
                self.assertEqual(FPR(tn,fp,fn,tp), o)
    
    def test_FDR(self):
        outputs = [0, 0.5, 1, 0, 0, 0, 1]
        for i,o in zip(self.inputs, outputs):
            tn,fp,fn,tp = i
            with self.subTest():
                self.assertEqual(FDR(tn,fp,fn,tp), o)


if __name__ == "__main__":
    unittest.main()
