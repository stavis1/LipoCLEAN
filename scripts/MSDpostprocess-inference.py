# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 12:18:10 2023

@author: Administrator
"""

import re
from functools import cache
import argparse

import dill
import pandas as pd
import numpy as np
from brainpy import isotopic_variants
from scipy.optimize import least_squares
import statsmodels.api as sm

###### input data
parser = argparse.ArgumentParser(
                    prog='MS-Dial lipid postprocessor inferenece',
                    description='Filters MS-Dial putative lipid identifications into good, bad, and requires manual reanalysis bins.')
parser.add_argument('-i', '--input', action = 'store', required = True,
                    help='msp output from MS-Dial. Multiple files from the same experiment can be input, e.g. positive and negative mode.')
parser.add_argument('-r', '--min_rt', action = 'store', required = False, default = 0.0, type = float,
                    help='Minimum observed retention time in minutes, used to filter peaks eluting in the dead volume. Default = 0.')
parser.add_argument('-l', '--cutoff_low', action = 'store', required = False, default = 0.2, type = float,
                    help='Putative IDs with a final model score below this value are labeled bad IDs. Default = 0.2.')
parser.add_argument('-t', '--cutoff_high', action = 'store', required = False, default = 0.8, type = float,
                    help='Putative IDs with a final model score above this value are labeled good IDs. Default = 0.8.')
parser.add_argument('-m', '--model', action = 'store', required = False, default = '/model.dill',
                    help='Pickled random forest model file created by training script. In the docker version if the file is not provided the default model will be used.')
parser.add_argument('-p', '--plots', action = 'store_true', required = False,
                    help='Generate plots to troubleshoot poor predictions. Default is no plots.')
parser.add_argument('-o', '--out_dir', action = 'store', required = True,
                    help='Directory for all outputs to be written to.')
args = parser.parse_args()

with open(args.model, 'rb') as pkl:
    pre_model, model = dill.load(pkl)


cut_high = args.cutoff_high #above this score lipid IDs are classed as good
cut_low = args.cutoff_low #below this score lipid IDs are classed as bad

###### function and object setup
#predicts the unit normalized intensities of the first three isotopic peaks for a particular formula
@cache
def iso_packet(formula):
    parsed = re.findall(r'\D\d*',formula)
    composition = {p[0]:int(p[1:]) if len(p) > 1 else 1 for p in parsed}
    return [p.intensity for p in isotopic_variants(composition, npeaks = 3)]

#mean square error of the observed isotope packet given the expectation from the formula
def iso_mse(observed, expected):
    obs = np.asarray([int(i) for i in re.findall(r':(\d+)',observed)])
    exp = np.asarray(expected)
    results = least_squares(lambda x: np.mean((obs/x - exp)**2), x0 = sum(obs), bounds = (1e-9, np.inf))
    return np.mean((obs/results.x - exp)**2)

###### initial data processing
lipid_data = []
for msd_file in args.input.split(','):
    lipid_data.append(pd.read_csv(msd_file, skiprows = 4, sep = '\t'))
lipid_data = pd.concat(lipid_data, ignore_index=True)

#first pass filter for extremely low confidence IDs
bad_idx = []
#must have MS2 data for ID, matches based on RT and exact mass are not considered valid
bad_idx.extend(lipid_data[[not(mz and rt) for mz,rt in zip(lipid_data['m/z matched'],lipid_data['RT matched'])]].index)
#MS-Dial annotates some features that don't have a formula, we do not consider this an ID
bad_idx.extend(lipid_data[[(('D' in f) or ('i' in f)) if type(f) == str else True for f in lipid_data['Formula']]].index) #this breaks brainpy
#features with an unknown ontology are also not considered IDs
bad_idx.extend(lipid_data[[type(o) != str or o in ['Unknown','Others'] for o in lipid_data['Ontology']]].index)
#features eluting in the dead volume are not considered reliably identifiable in our data and are discarded
#this is an optional runtime argument if unused min_rt = 0 and this line does nothing
bad_idx.extend(lipid_data[[r < args.min_rt for r in lipid_data['Average Rt(min)']]].index)
#these columns need to have valid numbers in them for the model to work
num_cols = ['Reference m/z','Average Mz','Dot product', 'S/N average']
bad_idx.extend(lipid_data[np.any(np.logical_not(np.isfinite(lipid_data[num_cols].to_numpy())), axis = 1)].index)
bad_idx.extend(lipid_data[[type(s) != str for s in lipid_data['MS1 isotopic spectrum']]].index)
bad_idx = list(set(bad_idx))
print(f'{len(bad_idx)} entries not considered\n', flush = True)

bad_rows = lipid_data.loc[bad_idx]
bad_rows.to_csv(f'{args.out_dir}/not_considered.tsv', sep = '\t', index = False)
lipid_data.drop(bad_idx, inplace=True)

#calculate predictors
lipid_data['mz_error'] = (lipid_data['Reference m/z'] - lipid_data['Average Mz'])/lipid_data['Average Mz']
lipid_data['iso_mse'] = [iso_mse(o,iso_packet(f)) for o,f in zip(lipid_data['MS1 isotopic spectrum'],lipid_data['Formula'])]

#initial prediction of 
predictor_cols = ['Dot product', 'S/N average', 'iso_mse', 'mz_error']
lipid_data['prepred'] = pre_model.predict(lipid_data[predictor_cols])
prepred = lipid_data[lipid_data['prepred'] == 1]

#lowess regression model predicts the referecene RT from the observed mean RT
#using only the initially high confidence lipids
#the residuals of this regression are used as a predictor for the final model
lowess = sm.nonparametric.lowess
lipid_data['pred_rt'] = lowess(prepred['Reference RT'],
                               prepred['Average Rt(min)'],
                               frac = 0.1, it = 3,
                               xvals = lipid_data['Average Rt(min)'])
lipid_data['rt_error'] = lipid_data['Reference RT'] - lipid_data['pred_rt']

#predict class probabilities, these will be used to bin IDs
predictor_cols =  ['Dot product', 'S/N average', 'iso_mse', 'mz_error', 'rt_error']
lipid_data['score'] = model.predict_proba(lipid_data[predictor_cols])[:,1]

#write outputs
lipid_data['pred_label'] = [1 if s > cut_high else 0 if s < cut_low else -1 for s in lipid_data['score']]
lipid_data[lipid_data['pred_label'] == 1].to_csv(f'{args.out_dir}/good_lipids.tsv', 
                                                               sep = '\t', index = False)
lipid_data[lipid_data['pred_label'] == -1].to_csv(f'{args.out_dir}/reanalyze_lipids.tsv', 
                                                                               sep = '\t', index = False)
lipid_data[lipid_data['pred_label'] == 0].to_csv(f'{args.out_dir}/bad_lipids.tsv', 
                                                              sep = '\t', index = False)

if args.plots:
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from itertools import combinations
    
    os.mkdir(f'{args.out_dir}/inference_QC')
    
    def get_colors(vals):
        low = min(vals)
        high = max(vals)
        return [cm.plasma(int(((val-low)/(high-low))*cm.plasma.N)) for val in vals]

    def get_sm(vals):
        colormap = matplotlib.colormaps['plasma']
        sm = plt.cm.ScalarMappable(cmap=colormap)
        sm.set_clim(vmin = min(vals), vmax = max(vals))
        return sm
    
    #plot retention time regression
    pts = sorted(list(zip(lipid_data['Average Rt(min)'], lipid_data['pred_rt'])), key = lambda x: x[0])
    
    fig, ax = plt.subplots()
    ax.scatter(lipid_data[lipid_data['prepred'] == 1]['Average Rt(min)'],
               lipid_data[lipid_data['prepred'] == 1]['Reference RT'],
                s =1 , c= 'k', marker = '.', label = 'in regression set')
    ax.scatter(lipid_data[lipid_data['prepred'] == 0]['Average Rt(min)'],
               lipid_data[lipid_data['prepred'] == 0]['Reference RT'],
                s =1 , c= 'r', marker = '.', label = 'not in regression set')
    ax.plot([p[0] for p in pts], [p[1] for p in pts],
            '-b', linewidth = 0.5, alpha = 0.5, label = 'regression')
    ax.legend()
    y0,y1 = ax.get_ylim()
    x0,x1 = ax.get_xlim()
    ax.set_aspect(abs(x1-x0)/abs(y1-y0))
    ax.set_ylabel('Reference RT')
    ax.set_xlabel('Observed RT')
    fig.savefig(f'{args.out_dir}/inference_QC/RT_alignment.svg', 
                dpi = 1000, bbox_inches = 'tight')
    
    log_pred = {'Dot product':False, 'S/N average':True, 'iso_mse':True, 'mz_error':False, 'rt_error':False}

    colors = get_colors(lipid_data['score'])
    sm = get_sm(lipid_data['score'])
    for pair in combinations(predictor_cols, 2):
        fig, ax = plt.subplots(figsize = (6,6))
        ax.scatter(lipid_data[pair[0]], lipid_data[pair[1]],
                   s = 1, color = colors, marker = '.')
        if log_pred[pair[0]]:
            ax.set_xscale('log')
        elif log_pred[pair[1]]:
            ax.set_yscale('log')
        ax.set_facecolor('lightgrey')
        ax.set_ylabel(pair[1])
        ax.set_xlabel(pair[0])
        clb = fig.colorbar(sm, ax = ax, location = 'right')
        clb.set_label('Score')
        fig.savefig(f'{args.out_dir}/inference_QC/{pair[0].replace("/","")}-{pair[1].replace("/","")}.svg', 
                    dpi = 1000, bbox_inches = 'tight')
        plt.close('all')

    
    #individual predictors correlation with final scores
    for predictor in predictor_cols:
        fig, ax = plt.subplots(figsize = (6,6))
        ax.scatter(lipid_data[predictor],
                   lipid_data['score'],
                   s = 1, c = 'k', marker = '.')
        if log_pred[predictor]:
            ax.set_xscale('log')
        ax.set_ylabel('Score')
        ax.set_xlabel(predictor)
        fig.savefig(f'{args.out_dir}/inference_QC/{predictor.replace("/","")}.svg', bbox_inches = 'tight')
