#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:31:18 2024

@author: 4vt
"""
import shelve
import re
from functools import cache

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as GBC
from brainpy import isotopic_variants
import statsmodels.api as sm

class model():
    def __init__(self, args):
        self.model = re.search(r"\.([^.]+)'>\Z", str(self.__class__)).group(1)
        self.cutoff = args.cutoffs[self.model]
        self.features = args.features[self.model]
        self.db = args.model
        self.probs = {}
        self.labels = {}
        self.calls = {}
        self.logs = args.logs
    
    def load(self):
        with shelve.open(self.db) as db:
            db[self.model] = self

    def dump(self):
        with shelve.open(self.db) as db:
            db[self.model] = self    
    
    def _predict_prob(self, data):
        probs = self.classifier.predict_proba(data[self.features])
        idx = list(self.classifier.classes_).index(1)
        probs = probs[:,idx]
        return probs
    
    def fit(self, data):
        self.classifier = GBC(n_estimators = 40)
        self.classifier.fit(data[self.features], 
                            data['label'])
        self.logs.info(f'Fit {self.model}')
    
    def assess(self, data, tag):
        self.probs[tag] = self._predict_prob(data)
        self.calls[tag] = self.predict(data)
        self.labels[tag] = data['label']
    
class prelim_model(model):   
    def predict(self, data):
        preds = self._predict_prob(data)
        preds = preds > self.cutoff
        return preds

class mz_correction(prelim_model):
    def __init__(self, args):
        super().__init__(args)
        self.ppm = args.ppm
        self.save_data = args.QC_plots

    def correct_data(self, data):
        #identify high confidence subset for correction
        calls = self.predict(data)
        subset = data[calls]
        
        #calculate observed m/z error
        mz_cols = np.array([c for c in subset.columns if c.startswith('observed_mz_')])
        mz_exp = subset['Reference m/z'].to_numpy()
        mz_error = subset[mz_cols].to_numpy() - mz_exp[:,np.newaxis]
        
        #save initial m/z values for QC plotting
        if self.save_data:
            all_error = data[mz_cols].to_numpy() - data['Reference m/z'].to_numpy()[:,np.newaxis]
            self.mz_initial = {c:all_error[np.isfinite(data[c]),i] for i,c in enumerate(mz_cols)}
        
        #calculate per-file midmeans of m/z error
        quartiles = np.nanquantile(mz_error, (0.25,0.75), axis = 0)
        mask = np.logical_or(mz_error < quartiles[0,:][np.newaxis,:],
                             mz_error > quartiles[1,:][np.newaxis,:])
        mz_error[mask] = np.full(mz_error.shape, np.nan)[mask]
        midmeans = np.nanmean(mz_error, axis = 0)
        mask = np.isnan(midmeans)
        if np.any(mask):
            print('Some m/z columns had no preliminary confident values. No m/z correction will be applied to these columns:')
            print('\n'.join(mz_cols[mask]))
            midmeans[mask] = np.zeros(midmeans.shape)[mask]
        
        #calculate corrected m/z error
        mz_exp = data['Reference m/z'].to_numpy()
        corr_mz = data[mz_cols].to_numpy() - midmeans[np.newaxis,:]
        mz_error = corr_mz - mz_exp[:,np.newaxis]

        #save corrected m/z values for QC plotting
        if self.save_data:
            self.mz_corrected = {c:mz_error[np.isfinite(data[c]),i] for i,c in enumerate(mz_cols)}
        
        #add mean error as a predictor to data
        mz_error = np.nanmean(mz_error, axis = 1)
        if self.ppm:
            mz_error = (mz_error/data['Reference m/z'])*1e6
        data['mz_error'] = mz_error
        
        self.logs.info('m/z correction has been applied.')
        return data
    
class rt_correction(prelim_model):
    def __init__(self, args):
        super().__init__(args)
        self.lowess_frac = args.lowess_frac
        self.rt_predictions = {}
        self.rt_expected = {}
        self.rt_observed = {}
        self.rt_calls = {}
    
    def correct_data(self, data):
        #identify high confidence subset for correction
        data['call'] = self.predict(data)
        
        #build lowess regression
        def rt_error(subset):
            lowess = sm.nonparametric.lowess
            regression = lowess(subset[subset['call']]['Reference RT'],
                                subset[subset['call']]['Average Rt(min)'],
                                frac = self.lowess_frac,
                                it = 3,
                                xvals = subset['Average Rt(min)'])
            rt_error = subset['Reference RT'] - regression
            #record regression for QC purposes
            file = next(f for f in subset['file'])
            self.rt_observed[file] = subset['Average Rt(min)']
            self.rt_expected[file] = subset['Reference RT']
            self.rt_predictions[file] = regression
            self.rt_calls[file] = subset['call']
            return rt_error
        
        data['rt_error'] = data.groupby('file').apply(rt_error).T
        data = data.drop(columns = ['call'])
        
        self.logs.info('RT correction has been applied.')
        return data

class predictor_model(model):
    def predict(self, data):
        preds = self._predict_prob(data)
        classes = [0 if p < self.cutoff[0] else 1 if p > self.cutoff[1] else -1 for p in preds]
        return classes

    def classify(self, data):
        data['class'] = self.predict(data)
        self.logs.info('Final classification is complete.')
        return data
    

@cache
def expected_isopacket(formula, npeaks):
    elms = re.findall(r'([A-Z][a-z]?)(\d+)', formula)
    composition = {e[0]:int(e[1]) for e in elms}
    isopacket = isotopic_variants(composition, npeaks = npeaks)
    intensities = np.array([p.intensity for p in isopacket])
    return intensities

def isotope_error(expected, observed):
    obs_intensities = np.array([float(o.split(':')[1]) for o in observed.split()])
    obs_intensities = obs_intensities/np.sum(obs_intensities)
    return np.nanmean(np.square(expected - obs_intensities))

def add_isotope_error(data):
    npeaks = len(list(data['MS1 isotopic spectrum'])[0].split())
    expected = [expected_isopacket(f, npeaks) for f in data['Formula']]
    data['isotope_error'] = [isotope_error(exp, obs) for exp, obs in zip(expected, data['MS1 isotopic spectrum'])]
    return data

