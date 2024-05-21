#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:31:18 2024

@author: 4vt
"""
# import shelve
import re
from functools import cache

import dill
import numpy as np
import pandas as pd
from brainpy import isotopic_variants
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from lineartree import LinearForestClassifier
from sklearn.pipeline import Pipeline


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
        with open(f'{self.db}/{self.model}', 'rb') as dillfile:
            self.classifier = dill.load(dillfile)

    def dump(self):
        with open(f'{self.db}/{self.model}', 'wb') as dillfile:
            dill.dump(self.classifier,dillfile)
        

    
    def assess(self, data, tag):
        self.probs[tag] = self._predict_prob(data)
        self.calls[tag] = self.predict(self.probs[tag])
        self.labels[tag] = data['label']
    
class prelim_model(model):   
    def preprocess(self, data):
        raise NotImplementedError()
    
    def fit(self, data):
        data = data.copy()
        data = data[np.isfinite(data['label'])]
        data = self.preprocess(data.copy())
        self.classifier = LogisticRegression(solver = 'liblinear')
        self.classifier.fit(data[self.features], data['label'])
        self.logs.info(f'Fit {self.model}')

    def _predict_prob(self, data):
        data = self.preprocess(data.copy())
        probs = self.classifier.predict_proba(data[self.features])
        idx = list(self.classifier.classes_).index(1)
        probs = probs[:,idx]
        return probs

    def predict(self, preds):
        preds = preds > self.cutoff
        self.logs.debug(f'{self.model} predicted {np.sum(preds)} positive out of {len(preds)} elements')
        return preds

class mz_correction(prelim_model):
    def __init__(self, args):
        super().__init__(args)
        self.ppm = args.ppm
        self.save_data = args.QC_plots
    
    def preprocess(self, data):
        processors = {'isotope_error': lambda x: np.log(x)}
        for column in processors:
            data[column] = processors[column](data[column])
        return data
    
    def calc_error(self, subset):
        mz_cols = np.array([c for c in subset.columns if c.startswith('observed_mz_')])
        mz_exp = subset['Reference m/z'].to_numpy()
        mz_error = subset[mz_cols].to_numpy() - mz_exp[:,np.newaxis]
        return (mz_cols, mz_error)
    
    def calc_midmeans(self, mz_error, mz_cols):
        quartiles = np.nanquantile(mz_error, (0.25,0.75), axis = 0)
        mask = np.logical_or(mz_error < quartiles[0,:][np.newaxis,:],
                             mz_error > quartiles[1,:][np.newaxis,:])
        mz_error[mask] = np.full(mz_error.shape, np.nan)[mask]
        midmeans = np.nanmean(mz_error, axis = 0)
        mask = np.isnan(midmeans)
        if np.any(mask):
            msg = 'Some m/z columns had no preliminary confident values. '
            msg += 'No m/z correction will be applied to these columns:\n'
            msg += '\n'.join(mz_cols[mask])
            self.logs.warning(msg)
            midmeans[mask] = np.zeros(midmeans.shape)[mask]
        
        for file, midmean in zip(mz_cols, midmeans):
            self.logs.debug(f'{file} had an m/z correction of {midmean} m/z')
        return midmeans
    
    def correct_error(self, data, mz_cols, midmeans):
        mz_exp = data['Reference m/z'].to_numpy()
        corr_mz = data[mz_cols].to_numpy() - midmeans[np.newaxis,:]
        mz_error = corr_mz - mz_exp[:,np.newaxis]
        return mz_error

    def correct_data(self, data):
        #identify high confidence subset for correction
        scores = self._predict_prob(data)
        calls = self.predict(scores)
        subset = data[calls]
        
        #calculate observed m/z error
        mz_cols, mz_error = self.calc_error(subset)
        
        #save initial m/z values for QC plotting
        if self.save_data:
            all_error = data[mz_cols].to_numpy() - data['Reference m/z'].to_numpy()[:,np.newaxis]
            self.mz_initial = {c:all_error[np.isfinite(data[c]),i] for i,c in enumerate(mz_cols)}
        
        #calculate per-file midmeans of m/z error
        midmeans = self.calc_midmeans(mz_error, mz_cols)
        
        #calculate corrected m/z error
        mz_error = self.correct_error(data, mz_cols, midmeans)

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

    def preprocess(self, data):
        processors = {'isotope_error': lambda x: np.log(x),
                      'mz_error':lambda x: np.abs(x)}
        for column in processors:
            data[column] = processors[column](data[column])
        return data

    def rt_error(self, subset):
        lowess = sm.nonparametric.lowess
        regression = lowess(subset[subset['call']]['Reference RT'],
                            subset[subset['call']]['Average Rt(min)'],
                            frac = self.lowess_frac,
                            it = 3,
                            xvals = subset['Average Rt(min)'])
        rt_error = subset['Reference RT'].to_numpy() - regression
        #record regression for QC purposes
        file = next(f for f in subset['file'])
        self.rt_observed[file] = subset['Average Rt(min)']
        self.rt_expected[file] = subset['Reference RT']
        self.rt_predictions[file] = regression
        self.rt_calls[file] = subset['call']
        self.logs.debug(f'RT regression fit for {file} used {np.sum(subset["call"])} observations')
        return rt_error    

    def correct_data(self, data):
        #identify high confidence subset for correction
        scores = self._predict_prob(data)
        data['call'] = self.predict(scores)
        
        #build lowess regressions
        rt_error = data.groupby('file')[data.columns].apply(self.rt_error)
        data['rt_error'] = [val for file in rt_error for val in file]
        data = data.drop(columns = ['call'])
        
        self.logs.info('RT correction has been applied.')
        return data

class predictor_model(model):
    def fit_preprocessor(self, data):
        self.preprocessor = StandardScaler()
        self.preprocessor.fit(data[self.features])

    def preprocess(self, data):
        processed = pd.DataFrame(self.preprocessor.transform(data[self.features]))
        if 'label' in data.columns:
            processed['label'] = data['label']
        return processed

    def fit(self, data):
        data = data.copy()
        data = data[np.isfinite(data['label'])]       
        LFclassifier = LinearForestClassifier(base_estimator=LinearRegression(),
                                              n_estimators = 10,
                                              max_depth = 10,
                                              max_features = 5)
        
        self.classifier = Pipeline([('scalar', StandardScaler()),
                                    ('classifier', LFclassifier)])
        self.classifier.fit(data[self.features], data['label'])
    
    def _predict_prob(self, data):
        probs = self.classifier.predict_proba(data[self.features])
        idx = list(self.classifier.classes_).index(1)
        probs = probs[:,idx]
        return probs
    
    def predict(self, preds):
        classes = [0 if p < self.cutoff[0] else 1 if p > self.cutoff[1] else -1 for p in preds]
        self.logs.debug(f'{self.model} predicted {np.sum(preds > self.cutoff[1])} positive out of {len(preds)} elements')
        return classes

    def classify(self, data):
        data['score'] = self._predict_prob(data)
        data['class'] = self.predict(data['score'])
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

