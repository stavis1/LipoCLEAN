#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:31:18 2024

@author: 4vt
"""
import shelve
import re

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as GBC

class model():
    def __init__(self, args):
        model = re.search(r'\.([^.]+)\Z', self.__class__)
        self.cutoff = args.cutoffs[model]
        self.features = args.features[model]
        self.db = args.model
    
    def load(self):
        with shelve.open(self.db) as db:
            db[self.__name__] = self

    def dump(self, args):
        with shelve.open(self.db) as db:
            db[self.__name__] = self    
    

class prelim_model(model):   
    def predict(self, data):
        preds = self.predict_proba(data)
        return preds > self.cutoff

class mz_correction(prelim_model):
    def fit(self, data):
        self.classifier = GBC(n_estimators = 40)
        self.calssifier.fit(data[self.features], 
                            data['label'])
    
    def correct_data(self, data):
        calls = self.predict(data)
        subset = data[calls]
        mz_cols = [c for c in subset.columns if c.startswith('observed_mz_')]
        mz_exp = subset['Reference m/z'].to_numpy()
        mz_err = subset[mz_cols].to_numpy() - mz_exp[:,np.newaxis]
        
        

class rt_correction(prelim_model):
    pass

class predictor_model(model):
    pass

def isotope_error(data):
    return

