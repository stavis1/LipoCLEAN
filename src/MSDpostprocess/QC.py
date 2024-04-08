#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:43:23 2024

@author: 4vt
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
import numpy as np

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

def plot_mz_QC(mz_model, args):
    filetypes = args.QC_plot_extensions
    
    raw_vals = mz_model.mz_initial
    cor_vals = mz_model.mz_corrected
    #all values together
    mz_error_histograms([v for vals in raw_vals.values() for v in vals], 
                        [v for vals in cor_vals.values() for v in vals], 
                        'm/z Error Correction', 
                        os.path.join(args.output, 'QC/mz_correction'), 
                        filetypes)
    
    # # per file plots
    # for file in raw_vals.keys():
    #     mz_error_histograms(raw_vals[file], 
    #                         cor_vals[file], 
    #                         f'{file} m/z Error Correction', 
    #                         os.path.join(args.output, f'QC/per_file_plots/{file}_mz_correction'), 
    #                         filetypes)
    
    #ROC plots
    if args.mode == 'train':
        for subset in mz_model.labels.keys():
            ROC(mz_model.probs[subset], 
                mz_model.labels[subset], 
                [mz_model.cutoff], 
                f'm/z Correction Model {subset}', 
                os.path.join(args.output, f'QC/mz_correction_model_{subset}'),
                filetypes)

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
    filetypes = args.QC_plot_extensions
    
    for dataset in rt_model.rt_observed.keys():
        plot_rt_correction(rt_model.rt_observed[dataset], 
                           rt_model.rt_expected[dataset], 
                           rt_model.rt_predictions[dataset], 
                           rt_model.rt_calls[dataset], 
                           dataset, 
                           os.path.join(args.output, f'QC/{dataset}_rt_regression'), 
                           filetypes)

    #ROC plots
    if args.mode == 'train':
        for subset in rt_model.labels.keys():
            ROC(rt_model.probs[subset], 
                rt_model.labels[subset], 
                [rt_model.cutoff], 
                f'RT Correction Model {subset}', 
                os.path.join(args.output, f'QC/rt_correction_model_{subset}'),
                filetypes)


def plot_final_QC(final_model, args):
    filetypes = args.QC_plot_extensions
    
    #ROC plots
    if args.mode == 'train':
        for subset in final_model.labels.keys():
            ROC(final_model.probs[subset], 
                final_model.labels[subset], 
                final_model.cutoff, 
                f'Final Model {subset}', 
                os.path.join(args.output, f'QC/final_model_{subset}'),
                filetypes)
