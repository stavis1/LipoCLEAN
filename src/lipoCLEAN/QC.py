#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:43:23 2024

@author: 4vt
"""
import os

import matplotlib
matplotlib.use('pdf')
matplotlib.use('svg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from collections import defaultdict

#settings
fsize = 12
ptsize = 15
size = (6,6)

#returns a list of colors based on some scalar value
def get_colors(vals):
    low = min(vals)
    high = max(vals)
    return [cm.plasma(int(((val-low)/(high-low))*cm.plasma.N)) for val in vals]

#this is used for creating a colorbar when plotting
def get_sm(vals):
    colormap = matplotlib.colormaps['plasma']
    sm = plt.cm.ScalarMappable(cmap=colormap)
    sm.set_clim(vmin = min(vals), vmax = max(vals))
    return sm

#True positive rate for ROC plots
def TPR(tn,fp,fn,tp):
    divisor = tp + fn
    if divisor == 0:
        return 0
    else:
        return tp/divisor

#False positive rate for ROC plots
def FPR(tn,fp,fn,tp):
    divisor = fp + tn
    if divisor == 0:
        return 0
    else:
        return fp/divisor

#False discovery rate for ROC plots
def FDR(tn,fp,fn,tp):
    divisor = fp + tp
    if divisor == 0:
        return 0
    else:
        return fp/divisor

def ROC(scores, labels, cutoffs, title, path, types):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_auc_score
    scores = scores[np.isfinite(labels)]
    labels = labels[np.isfinite(labels)]
    #calculate ROC curve
    tprs = [0]
    fprs = [0]
    for cut in sorted(list(scores), reverse=True):
        calls = [yhat >= cut for yhat in scores]
        tn,fp,fn,tp = confusion_matrix(labels, calls).flatten()
        tprs.append(TPR(tn,fp,fn,tp))
        fprs.append(FPR(tn,fp,fn,tp))
    tprs.append(1)
    fprs.append(1)
    
    #make base plot
    fig, ax = plt.subplots(figsize = size)
    ax.plot(fprs,tprs,'-k', linewidth = 1)
    ax.plot([0,1],[0,1], '--r', linewidth = 0.5)
    
    #annotate cutoffs
    for cutoff in cutoffs:
        calls = [yhat >= cutoff for yhat in scores]
        tn,fp,fn,tp = confusion_matrix(labels, calls).flatten()
        tpr = TPR(tn,fp,fn,tp)
        fpr = FPR(tn,fp,fn,tp)
        ax.scatter(fpr, tpr, s = ptsize, color = 'r', marker = '.')
        fdr = FDR(tn,fp,fn,tp)
        ax.text(fpr, tpr, f'cutoff: {cutoff}\nFDR: {"%.2f"%(fdr)}\nrecall: {"%.2f"%(tpr)}', 
                ha = 'left', va = 'top', fontsize = fsize)
    
    #format plot
    y0,y1 = ax.get_ylim()
    x0,x1 = ax.get_xlim()
    ax.set_aspect(abs(x1-x0)/abs(y1-y0))
    ax.set_ylim(-.001,1.001)
    ax.set_xlim(-.001,1.001)
    ax.set_facecolor('lightgrey')
    ax.set_ylabel('True Positive Rate', fontsize = fsize)
    ax.set_xlabel('False Positive Rate', fontsize = fsize)
    aucroc = roc_auc_score(labels, scores)
    ax.annotate(f'AUC: {"%.2f"%(aucroc)}', (0.5,0.5), ha='left', va='top', fontsize = fsize)
    ax.set_title(f'{title} Set ROC')
    for filetype in types:
        fig.savefig(f'{path}.{filetype}', bbox_inches = 'tight', dpi = 500)
    plt.close('all')
    
def mz_error_histograms(raw, corrected, title, path, types):
    allvals = list(raw) + list(corrected)
    bins = np.linspace(min(allvals), max(allvals), 80)

    fig, ax = plt.subplots(figsize = size)
    ax.hist(raw, bins = bins, color = 'r', alpha = 0.5, label = 'Raw m/z Error')
    ax.hist(corrected, bins = bins, color = 'k', alpha = 0.5, label = 'Corrected m/z Error')
    ax.legend()
    ax.set_ylabel('Number of m/z Values')
    ax.set_xlabel('Error (Da)')
    ax.set_title(title)
    for filetype in types:
        fig.savefig(f'{path}.{filetype}', bbox_inches = 'tight', dpi = 500)
    plt.close('all')

def TF_histogram(scores, labels, cutoffs, title, path, types):
    scores = scores[np.isfinite(labels)]
    labels = labels[np.isfinite(labels)]
    bins = np.linspace(min(scores), max(scores), 50)
    fig, ax = plt.subplots()
    ax.hist(scores[labels == 1], bins = bins, color = 'k', alpha = 0.5, label = 'True')    
    ax.hist(scores[labels == 0], bins = bins, color = 'r', alpha = 0.5, label = 'False')    
    ylim = ax.get_ylim()
    for cutoff in (cutoffs if hasattr(cutoffs, '__iter__') else [cutoffs]):
        ax.plot([cutoff]*2, ylim, '-k', linewidth = 1)
    ax.set_ylim(ylim)
    ax.set_xlabel('Model Score')
    ax.set_ylabel('Number of Observations')
    ax.set_title(title)
    ax.legend()
    for filetype in types:
        fig.savefig(f'{path}.{filetype}', bbox_inches = 'tight', dpi = 500)
    plt.close('all')

def write_metrics(calls, labels, scores, model, path):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_auc_score
    calls = np.array([c if c > 0 else 0 for c in calls])[np.isfinite(labels)]
    scores = scores[np.isfinite(labels)]
    labels = labels[np.isfinite(labels)]
    tn,fp,fn,tp = confusion_matrix(labels, calls).flatten()
    data = pd.DataFrame({'Model':model,
                         'FDR':FDR(tn, fp, fn, tp),
                         'FPR':FPR(tn, fp, fn, tp),
                         'Recall':TPR(tn, fp, fn, tp),
                         'AUCROC':roc_auc_score(labels, scores),
                         'True Positives':tp,
                         'False Positives':fp,
                         'False Negatives':fn,
                         'True Negatives':tn},
                        index = [0])
    header = not os.path.exists(path)
    data.to_csv(path, mode = 'a', header = header, index = False, sep = '\t')

def plot_mz_QC(mz_model, args):
    raw_vals = mz_model.mz_initial
    cor_vals = mz_model.mz_corrected
    #all values together
    mz_error_histograms([v for vals in raw_vals.values() for v in vals], 
                        [v for vals in cor_vals.values() for v in vals], 
                        'm/z Error Correction', 
                        os.path.join(args.output, 'QC/mz_correction'), 
                        args.QC_plot_extensions)
    
    # per file plots
    if args.QC_plots == 'all':
        for file in raw_vals.keys():
            mz_error_histograms(raw_vals[file], 
                                cor_vals[file], 
                                f'{file} m/z Error Correction', 
                                os.path.join(args.output, f'QC/per_file_plots/{os.path.basename(file)}_mz_correction'), 
                                args.QC_plot_extensions)
    
    #ROC plots
    if args.mode == 'train':
        for subset in mz_model.labels.keys():
            write_metrics(mz_model.calls[subset], 
                          mz_model.labels[subset], 
                          mz_model.probs[subset], 
                          f'mz_model {subset}', 
                          os.path.join(args.output, 'QC/metrics.tsv'))
            
            ROC(mz_model.probs[subset], 
                mz_model.labels[subset], 
                [mz_model.cutoff], 
                f'm/z Correction Model {subset}', 
                os.path.join(args.output, f'QC/mz_correction_model_{subset}'),
                args.QC_plot_extensions)

            TF_histogram(mz_model.probs[subset], 
                mz_model.labels[subset], 
                mz_model.cutoff, 
                f'Final Model {subset}', 
                os.path.join(args.output, f'QC/final_model_{subset}_histogram'),
                args.QC_plot_extensions)

    args.logs.info('Generated m/z QC plots')

def plot_rt_correction(observed, expected, predicted, calls, title, path, types):
    colors = ['k' if c else 'r' for c in calls]
    red_patch = mpatches.Patch(color='r', label='Not in Regression Set')
    black_patch = mpatches.Patch(color='k', label='In Regression Set')

    regression = np.array(sorted(zip(observed, predicted)))
    
    fig, ax = plt.subplots(figsize = size)
    ax.scatter(observed, expected, s = 1, c = colors, marker = '.')
    blue_line = ax.plot(regression[:,0], regression[:,1], '-b', label = 'Regression')
    ax.legend(handles=[red_patch,black_patch,blue_line[0]], loc = 'upper left')
    ax.set_ylabel('MS-DIAL Reference RT')
    ax.set_xlabel('Observed RT')
    ax.set_title(f'{title} RT Correction')
    for filetype in types:
        fig.savefig(f'{path}.{filetype}', bbox_inches = 'tight', dpi = 500)
    plt.close('all')

def plot_rt_QC(rt_model, args):
    for dataset in rt_model.rt_observed.keys():
        plot_rt_correction(rt_model.rt_observed[dataset], 
                           rt_model.rt_expected[dataset], 
                           rt_model.rt_predictions[dataset], 
                           rt_model.rt_calls[dataset], 
                           dataset, 
                           os.path.join(args.output, f'QC/{os.path.basename(dataset)}_rt_regression'), 
                           args.QC_plot_extensions)

    #ROC plots
    if args.mode == 'train':
        for subset in rt_model.labels.keys():
            write_metrics(rt_model.calls[subset], 
                          rt_model.labels[subset], 
                          rt_model.probs[subset], 
                          f'rt_model {subset}', 
                          os.path.join(args.output, 'QC/metrics.tsv'))
            
            ROC(rt_model.probs[subset], 
                rt_model.labels[subset], 
                [rt_model.cutoff], 
                f'RT Correction Model {subset}', 
                os.path.join(args.output, f'QC/rt_correction_model_{subset}'),
                args.QC_plot_extensions)

            TF_histogram(rt_model.probs[subset], 
                rt_model.labels[subset], 
                rt_model.cutoff, 
                f'Final Model {subset}', 
                os.path.join(args.output, f'QC/final_model_{subset}_histogram'),
                args.QC_plot_extensions)

    args.logs.info('Generated RT QC plots')

def plot_final_QC(final_model, args):
    #ROC plots
    if args.mode == 'train':
        for subset in final_model.labels.keys():
            write_metrics(final_model.calls[subset], 
                          final_model.labels[subset], 
                          final_model.probs[subset], 
                          f'final_model {subset}', 
                          os.path.join(args.output, 'QC/metrics.tsv'))
            
            ROC(final_model.probs[subset], 
                final_model.labels[subset], 
                final_model.cutoff, 
                f'Final Model {subset}', 
                os.path.join(args.output, f'QC/final_model_{subset}_ROC'),
                args.QC_plot_extensions)
            
            TF_histogram(final_model.probs[subset], 
                final_model.labels[subset], 
                final_model.cutoff, 
                f'Final Model {subset}', 
                os.path.join(args.output, f'QC/final_model_{subset}_histogram'),
                args.QC_plot_extensions)

def scores_plot(predictor1, predictor2, data, path, types):
    colors = get_colors(data['score'])
    sm = get_sm(data['score'])
    log_pred = defaultdict(lambda: False, 
                           {'Weighted dot product':False,
                            'Dot product':False, 
                            'S/N average':True, 
                            'isotope_error':True, 
                            'mz_error':False, 
                            'rt_error':False})

    fig, ax = plt.subplots(figsize = size)
    ax.scatter(data[predictor1], data[predictor2], 
               s = 1, c = colors, marker = '.')
    if log_pred[predictor1]:
        ax.set_xscale('log')
    if log_pred[predictor2]:
        ax.set_yscale('log')
    ax.set_facecolor('lightgrey')
    ax.set_xlabel(predictor1, fontsize = fsize)
    ax.set_ylabel(predictor2, fontsize = fsize)
    clb = fig.colorbar(sm, ax = ax, location = 'right')
    clb.set_label('Score', fontsize = fsize)
    for filetype in types:
        fig.savefig(f'{path}.{filetype}', bbox_inches = 'tight', dpi = 500)
    plt.close('all')

def plot_pairwise_scores(data, args):
    from itertools import combinations
    predictors = args.features['predictor_model']
    for pair in combinations(predictors, 2):
        name = f'{pair[0]}_{pair[1]}'.replace('/','-')
        scores_plot(pair[0], 
                    pair[1], 
                    data, 
                    os.path.join(args.output, f'QC/scores_plots/{name}'), 
                    args.QC_plot_extensions)
    
    args.logs.info('Generated final model QC plots')


