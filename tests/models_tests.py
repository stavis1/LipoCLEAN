#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 09:46:36 2024

@author: 4vt
"""
import sys
import os
import unittest
import re

import numpy as np
import pandas as pd

from MSDpostprocess.models import mz_correction, rt_correction, predictor_model, isotope_error, add_isotope_error
from MSDpostprocess.options import options
import tests

class mzModelTestSuite(tests.modelTestSuite):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_class = mz_correction
        self.get_features()
    
class mzCorrectionTestSuite(tests.hasWorkspaceTestSuite):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_class = mz_correction
        self.get_features()

    def get_features(self):
        init_argv = sys.argv
        test_path = os.path.dirname(__name__)
        sys.argv = [sys.argv[0], '--options', os.path.join(test_path, 'options.toml')]
        args = options()
        sys.argv = init_argv
        classname = str(self.model_class(args).__class__)
        self.model_name = re.search(r"\.([^.]+)'>\Z", classname).group(1)
        self.features = args.features[self.model_name]

    def make_mz_error_tools(self, N):
        rng = np.random.default_rng(1)
        
        #subclass mz_correction to modify the predict() method
        preds = rng.uniform(0,1,N)
        calls = preds > 0.1
        args = self.args
        args.cutoffs['test_mz_corr'] = args.cutoffs['mz_correction']
        args.features['test_mz_corr'] = args.features['mz_correction']
        args.ppm = True
        class test_mz_corr(mz_correction):
            def predict(self, data):
                return calls
        
        model = test_mz_corr(args)
        return (model, rng)
        
    def setUp(self):
        super().setUp()
        self.N = 200
        self.M = 16
        self.model, self.rng = self.make_mz_error_tools(self.N)
        
    def test_calc_error(self):
        mz_true = self.rng.lognormal(12, 1, self.N)

        #normal noise proportional to the intensity
        errs = self.rng.normal(0,1,(self.N,self.M))
        errs = errs*(mz_true[:,np.newaxis]*1e-6)
        mz_obs = mz_true[:,np.newaxis] + errs
        
        cols = [f'observed_mz_{i}' for i in range(self.M)]
        data = pd.DataFrame(mz_obs, columns = cols)
        data['Reference m/z'] = mz_true
        
        mz_cols, mz_error = self.model.calc_error(data)
        delta = np.nanmean(np.abs(errs - mz_error))
        
        with self.subTest():
            self.assertSequenceEqual(list(mz_cols), cols)
        with self.subTest():
            self.assertAlmostEqual(delta, 0)
    
    def test_calc_zero_midmeans(self):
        mz_true = self.rng.lognormal(12, 1, self.N)

        #normal noise proportional to the intensity
        errs = self.rng.normal(0,1,(self.N,self.M))
        errs = errs*(mz_true[:,np.newaxis]*1e-6)
        mz_obs = mz_true[:,np.newaxis] + errs
        
        cols = [f'observed_mz_{i}' for i in range(self.M)]
        data = pd.DataFrame(mz_obs, columns = cols)
        data['Reference m/z'] = mz_true
        
        mz_cols, mz_error = self.model.calc_error(data)
        midmeans = self.model.calc_midmeans(mz_error, mz_cols)
        delta = np.mean(np.abs(midmeans))
        self.assertAlmostEqual(delta, 0, delta = 0.01)

    def test_calc_nonzero_midmeans(self):
        mz_true = self.rng.lognormal(12, 1, self.N)

        #normal noise proportional to the intensity
        errs = self.rng.normal(0,1,(self.N,self.M))
        errs = errs*(mz_true[:,np.newaxis]*1e-6)
        mz_obs = mz_true[:,np.newaxis] + errs
        
        
        cols = [f'observed_mz_{i}' for i in range(self.M)]
        data = pd.DataFrame(mz_obs, columns = cols)
        data['Reference m/z'] = mz_true
        
        mz_cols, mz_error = self.model.calc_error(data)
        midmeans = self.model.calc_midmeans(mz_error, mz_cols)
        delta = np.mean(np.abs(midmeans))
        self.assertAlmostEqual(delta, 0, delta = 0.01)
    
    def test_remove_no_mz_error(self):
        mz_true = self.rng.lognormal(12, 1, self.N)
        mz_obs = mz_true[:,np.newaxis] + np.zeros((self.N,self.M))
        
        data = pd.DataFrame(mz_obs, columns = [f'observed_mz_{i}' for i in range(self.M)])
        data['Reference m/z'] = mz_true
        
        data = self.model.correct_data(data)
        
        mean_error = np.nanmean(data['mz_error'])
        self.assertAlmostEqual(mean_error, 0)

    def test_remove_mz_error(self):
        N = 200
        M = 16
        model, rng = self.make_mz_error_tools(N)
        
        #generate a fake dataset
        #we simulate an expected distribution of m/z values for a set of analytes
        mz_true = rng.lognormal(12, 1, N)
        
        #normal noise proportional to the intensity
        errs = rng.normal(0,1,(N,M))
        errs = errs*(mz_true[:,np.newaxis]*1e-6)
        mz_obs = mz_true[:,np.newaxis] + errs

        #simulate incorrect calls which result in large m/z errors
        extreme_mask = rng.uniform(0,1,N) < 0.1
        extreme_vals = rng.normal(0,1,N)
        mz_obs[extreme_mask] = (mz_obs + ((mz_obs*1e-3)*extreme_vals[:,np.newaxis]))[extreme_mask]
        
        #calculate true m/z error
        true_mz_err = np.nanmean(mz_true[:,np.newaxis] - mz_obs, axis = 1)
        
        #sample specific bias values that we wish to remove
        bias = rng.normal(0,1e-2,M)
        mz_obs = mz_obs + bias[np.newaxis, :]
        
        #simulate missing values
        null_mask = rng.uniform(0,1,(N,M)) < 0.1
        mz_obs[null_mask] = np.full((N,M), np.nan)[null_mask]
        
        data = pd.DataFrame(mz_obs, columns = [f'observed_mz_{i}' for i in range(M)])
        data['Reference m/z'] = mz_true
        
        data = model.correct_data(data)
        
        mean_error = np.nanmean((true_mz_err - data['mz_error'])[np.logical_not(extreme_mask)])
        self.assertAlmostEqual(mean_error, 0, delta = 0.01)
        
       
class rtModelTestSuite(tests.modelTestSuite):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_class = rt_correction
        self.get_features()

class rtCorrectionTestSuit(tests.hasWorkspaceTestSuite):
    def setUp(self):
        super().setUp()
        self.N = 200
        self.rng = np.random.default_rng(1)
        
        #subclass mz_correction to modify the predict() method
        args = self.args
        args.cutoffs['test_rt_corr'] = args.cutoffs['rt_correction']
        args.features['test_rt_corr'] = args.features['rt_correction']
        class test_rt_corr(rt_correction):
            def predict(self, data):
                rng = np.random.default_rng(1)
                preds = rng.uniform(0,1,data.shape[0])
                calls = preds > 0.1
                return calls
        
        self.model = test_rt_corr(args)
    
    def test_correct_no_drift_no_noise(self):
        ref = self.rng.uniform(0,10, self.N)
        obs = ref
        
        data = pd.DataFrame({'call':self.model.predict(np.zeros(self.N)),
                             'Reference RT':ref,
                             'Average Rt(min)':obs,
                             'file':[1]*self.N})
        
        rt_error = self.model.rt_error(data)
        delta = np.mean(np.abs(rt_error))
        self.assertAlmostEqual(delta, 0)

    def test_correct_drift_no_noise(self):
        ref = self.rng.uniform(0,10, self.N)
        obs = ref**2
        
        data = pd.DataFrame({'call':self.model.predict(np.zeros(self.N)),
                             'Reference RT':ref,
                             'Average Rt(min)':obs,
                             'file':[1]*self.N})
        
        rt_error = self.model.rt_error(data)
        delta = np.mean(np.abs(rt_error))
        self.assertLess(delta, 0.5)


    def test_correct_no_drift_noise(self):
        ref = self.rng.uniform(0,10, self.N)
        noise = self.rng.uniform(-0.5,0.5, self.N)
        obs = ref + noise
        
        data = pd.DataFrame({'call':self.model.predict(np.zeros(self.N)),
                             'Reference RT':ref,
                             'Average Rt(min)':obs,
                             'file':[1]*self.N})
        
        rt_error = self.model.rt_error(data)
        delta = np.mean(np.abs(rt_error - noise))
        self.assertLess(delta, 0.5)

    def test_correct_drift_noise(self):
        ref = self.rng.uniform(0,10, self.N)
        noise = self.rng.uniform(-0.5,0.5, self.N)
        obs = ref**2 + noise
        
        data = pd.DataFrame({'call':self.model.predict(np.zeros(self.N)),
                             'Reference RT':ref,
                             'Average Rt(min)':obs,
                             'file':[1]*self.N})
        
        rt_error = self.model.rt_error(data)
        delta = np.mean(np.abs(rt_error - noise))
        self.assertLess(delta, 0.5)


    def test_correct_drift_noise_multiple_files(self):
        noises = []
        data = []
        for i in range(5):
            ref = self.rng.uniform(0,10, self.N)
            noise = self.rng.uniform(-0.5,0.5, self.N)
            obs = ref**2 + noise
            noises.extend(noise)
            
            tmp = pd.DataFrame({'call':self.model.predict(np.zeros(self.N)),
                                'Reference RT':ref,
                                'Average Rt(min)':obs,
                                'file':[i]*self.N})
            data.append(tmp)
        
        data = pd.concat(data)
        data = self.model.correct_data(data)
        delta = np.mean(np.abs(data['rt_error'].to_numpy() - np.array(noises)))

        self.assertLess(delta, 0.5)
        

class finalModelTestSuite(tests.modelTestSuite):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_class = predictor_model
        self.get_features()


if __name__ == "__main__":
    unittest.main()

