# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 12:18:10 2023

@author: Administrator
"""

import re
from functools import cache
import argparse

###### input data
parser = argparse.ArgumentParser(
                    prog='MS-Dial lipid postprocessor inferenece',
                    description='Filters MS-Dial putative lipid identifications into good, bad, and requires manual reanalysis bins.')
parser.add_argument('-i', '--input', action = 'store', required = True,
                    help='msp output from MS-Dial. Multiple files from the same experiment can be input, e.g. positive and negative mode.')
parser.add_argument('-r', '--min_rt', action = 'store', required = False, default = 0.0, type = float,
                    help='Minimum observed retention time in minutes, used to filter peaks eluting in the dead volume. Default = 0.')
parser.add_argument('-l', '--cutoff_low', action = 'store', required = False, default = 0.3, type = float,
                    help='Putative IDs with a final model score below this value are labeled bad IDs. Default = 0.2.')
parser.add_argument('-t', '--cutoff_high', action = 'store', required = False, default = 0.8, type = float,
                    help='Putative IDs with a final model score above this value are labeled good IDs. Default = 0.8.')
parser.add_argument('-m', '--model', action = 'store', required = False, default = '/model.dill',
                    help='Pickled random forest model file created by training script. In the docker version if the file is not provided the default model will be used.')
parser.add_argument('-p', '--plots', action = 'store_true', required = False,
                    help='Generate plots to troubleshoot poor predictions. Default is no plots.')
parser.add_argument('-n', '--ppm', action = 'store_true', required = False,
                    help='Use m/z error in units of ppm. Default is Daltons.')
parser.add_argument('-o', '--out_dir', action = 'store', required = True,
                    help='Directory for all outputs to be written to.')
args = parser.parse_args()
cut_high = args.cutoff_high #above this score lipid IDs are classed as good
cut_low = args.cutoff_low #below this score lipid IDs are classed as bad


import dill
import pandas as pd
import numpy as np
from brainpy import isotopic_variants
from scipy.optimize import minimize
from scipy.optimize import Bounds
import statsmodels.api as sm

###### function and object setup
with open(args.model, 'rb') as pkl:
    mz_model, rt_model, model = dill.load(pkl)

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
    results = minimize(lambda x: np.mean((obs/x[0] - exp)**2), x0 = [sum(obs)], bounds = Bounds(lb = 1e-9))
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
nonnancols = ['Dot product', 'S/N average', 'Average Rt(min)', 'Reference m/z']
bad_idx.extend(lipid_data[[any(np.isnan(v)) for v in zip(*[lipid_data[c] for c in nonnancols])]].index)
bad_idx.extend(lipid_data[[type(s) != str for s in lipid_data['MS1 isotopic spectrum']]].index)
bad_idx = list(set(bad_idx))
print(f'{len(bad_idx)} entries not considered\n', flush = True)

bad_rows = lipid_data.loc[bad_idx]
bad_rows.to_csv(f'{args.out_dir}/not_considered.tsv', sep = '\t', index = False)
lipid_data.drop(bad_idx, inplace=True)

#identify columns containing m/z values
mz_cols = list(lipid_data.columns)[list(lipid_data.columns).index('MS/MS spectrum')+1:]
if any(lipid_data[c].dtype.kind != 'f' for c in mz_cols):
    print('Potentially invalid data types in m/z columns. This could be due to non m/z columns being inserted after "MS/MS spectrum". Please check', flush = True)
    print('m/z columns:\n' + '\n'.join(mz_cols))

#calculate predictors
lipid_data['iso_mse'] = [iso_mse(o,iso_packet(f)) for o,f in zip(lipid_data['MS1 isotopic spectrum'],lipid_data['Formula'])]

#initial prediction of good lipids for fitting the m/z correction
predictor_cols = ['iso_mse', 'Dot product', 'S/N average']
lipid_data['rt_prepred'] = mz_model.predict(lipid_data[predictor_cols])
mz_set = lipid_data[lipid_data['rt_prepred'] == 1]

#store initial m/z errors for plotting
if args.plots:
    init_deltas = lipid_data[mz_cols].to_numpy() - np.asarray([lipid_data['Reference m/z']]*len(mz_cols)).T
    init_deltas = init_deltas.flatten()

#calculate the midmean of m/z errors within the mz_model prefiltered set on a per file basis
mz_deltas = mz_set[mz_cols].to_numpy() - np.asarray([mz_set['Reference m/z']]*len(mz_cols)).T
if args.ppm:
    mz_deltas = (mz_deltas / np.asarray([np.nanmean(mz_set[mz_cols], axis = 1)]*mz_deltas.shape[1]).T) * 1e7
quartiles = np.nanquantile(mz_deltas, q = [0.75, 0.25], axis = 0)
midmeans = np.nanmean(mz_deltas, axis = 0, where = np.logical_and(np.less_equal(mz_deltas, [quartiles[0,:]]*mz_set.shape[0]),
                                                                  np.greater_equal(mz_deltas, [quartiles[1,:]]*mz_set.shape[0])))

#apply the correction
mz_deltas = lipid_data[mz_cols].to_numpy() - np.asarray([lipid_data['Reference m/z']]*len(mz_cols)).T
if args.ppm:
    mz_deltas = (mz_deltas / np.asarray([np.nanmean(lipid_data[mz_cols], axis = 1)]*mz_deltas.shape[1]).T) * 1e7
mz_deltas = mz_deltas - np.asarray([midmeans]*lipid_data.shape[0])
for i, col in enumerate(mz_cols):
    lipid_data[col] = mz_deltas[:, i]
lipid_data['mz_error'] = np.nanmean(lipid_data[mz_cols], axis = 1)

#final m/z errors for plotting
if args.plots:
    final_deltas = lipid_data[mz_cols].to_numpy().flatten()

#initial prediction of good lipids for fitting the RT alignment
predictor_cols = ['Dot product', 'S/N average', 'iso_mse', 'mz_error']
lipid_data['rt_prepred'] = rt_model.predict(lipid_data[predictor_cols])
rt_set = lipid_data[lipid_data['rt_prepred'] == 1]

#lowess regression model predicts the referecene RT from the observed mean RT
#using only the initially high confidence lipids
#the residuals of this regression are used as a predictor for the final model
lowess = sm.nonparametric.lowess
lipid_data['pred_rt'] = lowess(rt_set['Reference RT'],
                               rt_set['Average Rt(min)'],
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
    ax.scatter(lipid_data[lipid_data['rt_prepred'] == 1]['Average Rt(min)'],
               lipid_data[lipid_data['rt_prepred'] == 1]['Reference RT'],
                s =1 , c= 'k', marker = '.', label = 'in regression set')
    ax.scatter(lipid_data[lipid_data['rt_prepred'] == 0]['Average Rt(min)'],
               lipid_data[lipid_data['rt_prepred'] == 0]['Reference RT'],
                s =1 , c= 'r', marker = '.', label = 'not in regression set')
    ax.plot([p[0] for p in pts], [p[1] for p in pts],
            '-b', linewidth = 0.5, alpha = 0.5, label = 'regression')
    ax.legend()
    y0,y1 = ax.get_ylim()
    x0,x1 = ax.get_xlim()
    ax.set_aspect(abs(x1-x0)/abs(y1-y0))
    ax.set_ylabel('Reference RT')
    ax.set_xlabel('Observed RT')
    fig.savefig(f'{args.out_dir}/inference_QC/RT_alignment.PNG', 
                dpi = 1000, bbox_inches = 'tight')
    fig.savefig(f'{args.out_dir}/inference_QC/RT_alignment.svg', 
                bbox_inches = 'tight')

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
        fig.savefig(f'{args.out_dir}/inference_QC/{pair[0].replace("/","")}-{pair[1].replace("/","")}.png', 
                    dpi = 1000, bbox_inches = 'tight')
        fig.savefig(f'{args.out_dir}/inference_QC/{pair[0].replace("/","")}-{pair[1].replace("/","")}.svg', 
                    bbox_inches = 'tight')
        plt.close('all')

    
    #individual predictors correlation with final scores
    for predictor in predictor_cols:
        fig, ax = plt.subplots(figsize = (6,6))
        ax.scatter(lipid_data[predictor],
                   lipid_data['score'],
                   s = 1, c = 'k', marker = '.')
        xlim = [x if x > 0 else min(lipid_data[predictor]) for x in ax.get_xlim()]  if log_pred[predictor] else ax.get_xlim()
        ax.plot(xlim, [cut_low]*2, '-b', linewidth = .5)
        ax.plot(xlim, [cut_high]*2, '-b', linewidth = .5)
        if log_pred[predictor]:
            ax.set_xscale('log')
        ax.set_xlim(xlim)
        ax.set_ylabel('Score')
        ax.set_xlabel(predictor)
        fig.savefig(f'{args.out_dir}/inference_QC/{predictor.replace("/","")}.png', dpi = 1000, bbox_inches = 'tight')
        fig.savefig(f'{args.out_dir}/inference_QC/{predictor.replace("/","")}.svg', bbox_inches = 'tight')

    #score distribuiton
    fig, ax = plt.subplots()
    ax.hist(lipid_data['score'], bins = 80, color = 'k')
    ax.set_xlim(0,1)
    ylim = ax.get_ylim()
    ax.plot([cut_low]*2, ylim, '-b', linewidth = 0.5)
    ax.plot([cut_high]*2, ylim, '-b', linewidth = 0.5)
    ax.set_ylim(ylim)
    ax.set_xlabel('Final Model Scores')
    ax.set_ylabel('Count')
    fig.savefig(f'{args.out_dir}/inference_QC/ScoresDistribution.png', dpi = 1000, bbox_inches = 'tight')
    fig.savefig(f'{args.out_dir}/inference_QC/ScoresDistribution.svg', bbox_inches = 'tight')

    #plot m/z correction
    fig, ax = plt.subplots()
    ax.hist(init_deltas, bins = 100, color = 'r', alpha = 0.5, label = 'Uncorrected')
    ax.hist(final_deltas, bins = 100, color = 'k', alpha = 0.5, label = 'Corrected')
    ax.legend()
    ax.set_xlabel('Delta m/z')
    fig.savefig(f'{args.out_dir}/inference_QC/mz_correction.png', dpi = 1000, bbox_inches = 'tight')
    fig.savefig(f'{args.out_dir}/inference_QC/mz_correction.svg', bbox_inches = 'tight')

    #m/z error vs m/z
    fig, ax = plt.subplots()
    ax.scatter(lipid_data['Average Mz'], lipid_data['mz_error'], s = 1, c = colors, marker = '.')
    clb = fig.colorbar(sm, ax = ax, location = 'right')
    clb.set_label('Score')
    ax.set_ylabel('m/z Error')
    ax.set_ylabel('Average m/z')
    fig.savefig(f'{args.out_dir}/training_QC/mz_errorVmz.png', dpi = 1000, bbox_inches = 'tight')
    fig.savefig(f'{args.out_dir}/training_QC/mz_errorVmz.svg', bbox_inches = 'tight')

