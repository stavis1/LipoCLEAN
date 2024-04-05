#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:43:23 2024

@author: 4vt
"""

class QC_accumulator:
    def __init__(self):
        self.mz_model = {}
        self.mz_initial = {}
        self.mz_corrected = {}

        self.rt_model = {}
        self.rt_observed = {}
        self.rt_predicted = {}

        self.final_model = {}
