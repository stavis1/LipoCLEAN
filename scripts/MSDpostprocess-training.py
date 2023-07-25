# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 17:41:58 2023

@author: Administrator
"""

import os
import re
from functools import cache
import argparse
from itertools import combinations

import dill
import pandas as pd
import numpy as np
from brainpy import isotopic_variants
from scipy.optimize import least_squares
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

###### input data
parser = argparse.ArgumentParser(
                    prog='MS-Dial lipid postprocessor inferenece',
                    description='Filters MS-Dial putative lipid identifications into good, bad, and requires manual reanalysis bins.')
parser.add_argument('-i', '--input', action = 'store', required = True,
                    help='Modified msp output from MS-Dial, see README for details.')
parser.add_argument('-r', '--min_rt', action = 'store', required = False, default = 0.0, type = float,
                    help='Minimum observed retention time in minutes, used to filter peaks eluting in the dead volume.')
parser.add_argument('-l', '--cutoff_low', action = 'store', required = False, default = 0.2, type = float,
                    help='Putative IDs with a final model score below this value are labeled bad IDs. Default = 0.2')
parser.add_argument('-t', '--cutoff_high', action = 'store', required = False, default = 0.8, type = float,
                    help='Putative IDs with a final model score above this value are labeled good IDs. Default = 0.8')
parser.add_argument('-o', '--out_dir', action = 'store', required = True,
                    help='Directory for all outputs to be written to.')
args = parser.parse_args()

###### function and object setup
rng = np.random.default_rng(1234)

cut_high = args.cutoff_high #above this score lipid IDs are classed as good
cut_low = args.cutoff_low #below this score lipid IDs are classed as bad
#between the above two scores lipid IDs are set aside for manual reanalysis

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
    results = least_squares(lambda x: np.mean((obs - x*exp)**2), x0 = obs[0])
    return np.mean((obs - results.x*exp)**2)/obs[0]

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

os.mkdir(f'{args.out_dir}/training_QC')

###### initial data processing
#we allow training on multiple experiments
#these can have different elution profiles
lipid_data = []
for i,msd_file in enumerate(args.input.split(',')):
    lipid_data.append(pd.read_csv(msd_file, sep = '\t'))
    lipid_data[-1]['experiment'] = [os.path.basename(msd_file)]*lipid_data[-1].shape[0]
lipid_data = pd.concat(lipid_data, ignore_index=True)

#first pass filter for extremely low confidence IDs
bad_idx = []
#must have MS2 data for ID, matches based on RT and exact mass are not considered valid
bad_idx.extend(lipid_data[[not(mz and rt) for mz,rt in zip(lipid_data['m/z matched'],lipid_data['RT matched'])]].index)
#MS-Dial annotates some features that don't have a formula, we do not consider this an ID
bad_idx.extend(lipid_data[[type(f) != str for f in lipid_data['Formula']]].index)
bad_idx.extend(lipid_data[['D' in f or 'i' in f for f in lipid_data['Formula'] if type(f) == str]].index) #this breaks brainpy
#features with an unknown ontology are also not considered IDs
bad_idx.extend(lipid_data[[type(o) != str or o in ['Unknown','Others'] for o in lipid_data['Ontology']]].index)
#features eluting in the dead volume are not considered reliably identifiable in our data and are discarded
#this is an optional runtime argument if unused min_rt = 0 and this line does nothing
bad_idx.extend(lipid_data[[r < args.min_rt for r in lipid_data['Average Rt(min)']]].index)
bad_rows = lipid_data.loc[bad_idx]
bad_rows.to_csv(f'{args.out_dir}/training_QC/not_considered.tsv', sep = '\t', index = True)
lipid_data.drop(bad_idx, inplace=True)

#calculate predictors
lipid_data['mz_error'] = (lipid_data['Reference m/z'] - lipid_data['Average Mz'])/lipid_data['Average Mz']
lipid_data['iso_mse'] = [iso_mse(o,iso_packet(f)) for o,f in zip(lipid_data['MS1 isotopic spectrum'],lipid_data['Formula'])]

#pull out a subset of the data for testing at the end
test_idx = rng.choice(lipid_data.index, size = int(lipid_data.shape[0]*.15))
test_set = lipid_data.loc[test_idx]
lipid_data.drop(test_idx, inplace = True)
lipid_data['test_set'] = [False]*lipid_data.shape[0]
test_set['test_set'] = [True]*test_set.shape[0]

#initial prediction model for filtering for RT regression
predictor_cols = ['Dot product', 'S/N average', 'iso_mse', 'mz_error']
Y = lipid_data['label']
X = lipid_data[predictor_cols]
pre_model = GBC(n_estimators=40, max_features = 3).fit(X, Y)
print('trained first model', flush = True)

#predict class and filter down to initial positives for lowess RT correction model
lipid_data['prepred'] = pre_model.predict(lipid_data[predictor_cols])
test_set['prepred'] = pre_model.predict(test_set[predictor_cols])
prepred = lipid_data[lipid_data['prepred'] == 1]


#retention time alignment is done on an experiment-by-experiment basis
#average observed retention times are aligned against the reference value
pred_rts = {}
for i in set(lipid_data['experiment']):
    #lowess regression model predicts the referecene RT from the observed mean RT
    #using only the initially high confidence lipids
    #the residuals of this regression are used as a predictor for the final model
    temp_df = pd.concat([lipid_data[lipid_data['experiment'] == i],
                         test_set[test_set['experiment'] == i]])
    lowess = sm.nonparametric.lowess
    rt_preds = lowess(prepred[prepred['experiment'] == i]['Reference RT'],
                      prepred[prepred['experiment'] == i]['Average Rt(min)'],
                      frac = 0.1, it = 3,
                      xvals = temp_df['Average Rt(min)'])
    pred_rts.update({i:rt for i,rt in zip(temp_df.index, rt_preds)})

lipid_data['pred_rt'] = [pred_rts[i] for i in lipid_data.index]
lipid_data['rt_error'] = lipid_data['Reference RT'] - lipid_data['pred_rt']

test_set['pred_rt'] = [pred_rts[i] for i in test_set.index]
test_set['rt_error'] = test_set['Reference RT'] - test_set['pred_rt']
print('fit retention time corrections', flush = True)

#full prediction model for final class predictions
predictor_cols = ['Dot product', 'S/N average', 'iso_mse', 'mz_error', 'rt_error']
Y = lipid_data['label']
X = lipid_data[predictor_cols]
model = GBC(n_estimators=40, max_features = 4).fit(X, Y)
print('trained final model', flush = True)

#the model is trained so the test set can be safely reintegrated with the full data
lipid_data = pd.concat([lipid_data, test_set])

#predict class probabilities, these will be used to bin IDs
lipid_data['score'] = model.predict_proba(lipid_data[predictor_cols])[:,1]

#write GBC models to file for use with the inference script
with open(f'{args.out_dir}/model.dill', 'wb') as pkl:
    dill.dump((pre_model, model), pkl)

#### QC information to ensure that training went well
print('writing QC information')

#confusion matricies for the first round model
pre_test_confuse = pd.DataFrame(confusion_matrix(lipid_data[lipid_data['test_set']]['label'],
                                                 lipid_data[lipid_data['test_set']]['prepred']),
                                index = ['Predicted Bad', 'Predicted Good'],
                                columns = ['Bad', 'Good'])
pre_test_confuse.to_csv(f'{args.out_dir}/training_QC/test_set_confusion_matrix_pre_model.tsv', sep = '\t')

pre_train_confuse = pd.DataFrame(confusion_matrix(lipid_data[np.logical_not(lipid_data['test_set'])]['label'],
                                                  lipid_data[np.logical_not(lipid_data['test_set'])]['prepred']),
                                 index = ['Predicted Bad', 'Predicted Good'],
                                 columns = ['Bad', 'Good'])
pre_train_confuse.to_csv(f'{args.out_dir}/training_QC/train_set_confusion_matrix_pre_model.tsv', sep = '\t')

#confusion matricies for the final model
lipid_data['pred_label'] = [1 if s > cut_high else 0 if s < cut_low else -1 for s in lipid_data['score']]
test_confuse = pd.DataFrame(confusion_matrix(lipid_data[lipid_data['test_set']]['pred_label'],
                                                 lipid_data[lipid_data['test_set']]['label']),
                                index = ['Reanalyze', 'Predicted Bad', 'Predicted Good'],
                                columns = ['null', 'Bad', 'Good'])
test_confuse.pop('null')
test_confuse.to_csv(f'{args.out_dir}/training_QC/test_set_confusion_matrix_full_model.tsv', sep = '\t')

train_confuse = pd.DataFrame(confusion_matrix(lipid_data[np.logical_not(lipid_data['test_set'])]['pred_label'],
                                              lipid_data[np.logical_not(lipid_data['test_set'])]['label']),
                                index = ['Reanalyze', 'Predicted Bad', 'Predicted Good'],
                                columns = ['null', 'Bad', 'Good'])
train_confuse.pop('null')
train_confuse.to_csv(f'{args.out_dir}/training_QC/train_set_confusion_matrix_full_model.tsv', sep = '\t')

#write outputs
lipid_data[lipid_data['pred_label'] == 1].to_csv(f'{args.out_dir}/training_QC/good_lipids.tsv', 
                                                               sep = '\t', index = False)
lipid_data[lipid_data['pred_label'] == -1].to_csv(f'{args.out_dir}/training_QC/reanalyze_lipids.tsv', 
                                                                               sep = '\t', index = False)
lipid_data[lipid_data['pred_label'] == 0].to_csv(f'{args.out_dir}/training_QC/bad_lipids.tsv', 
                                                              sep = '\t', index = False)

#plot retention time regression
for exp in set(lipid_data['experiment']):
    lipids = lipid_data[[e == exp for e in lipid_data['experiment']]]
    pts = sorted(list(zip(lipids['Average Rt(min)'], lipids['pred_rt'])), key = lambda x: x[0])
    
    fig, ax = plt.subplots()
    df = lipids[[p == 1 for p in lipids['prepred']]]
    ax.scatter(df['Average Rt(min)'],
               df['Reference RT'],
               s =1 , c= 'k', marker = '.', label = 'in regression set')
    df = lipids[[p == 0 for p in lipids['prepred']]]
    ax.scatter(df['Average Rt(min)'],
               df['Reference RT'],
               s =1 , c= 'r', marker = '.', label = 'not in regression set')
    ax.plot([p[0] for p in pts], [p[1] for p in pts],
            '-b', linewidth = 0.5, alpha = 0.5, label = 'regression')
    ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    y0,y1 = ax.get_ylim()
    x0,x1 = ax.get_xlim()
    ax.set_aspect(abs(x1-x0)/abs(y1-y0))
    ax.set_ylabel('Reference RT')
    ax.set_xlabel('Observed RT')
    ax.set_title(f'Experiment {exp}')
    fig.savefig(f'{args.out_dir}/training_QC/RT_alignment-exp{exp}.svg', 
                bbox_inches = 'tight')
    plt.close('all')

log_pred = {'Dot product':False, 'S/N average':True, 'iso_mse':True, 'mz_error':False, 'rt_error':False}
#scatterplots of all possible pairs of predictors colored by the final model score
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
    fig.savefig(f'{args.out_dir}/training_QC/{pair[0].replace("/","")}-{pair[1].replace("/","")}_scores.svg', 
                bbox_inches = 'tight')
    plt.close('all')

#the same scatterplots as above but colored by each point's position in the confusion matrix
def confusion(label, prediction):
    if label:
        if prediction == 1:
            return 'True Positive'
        elif prediction == -1:
            return 'Reanalyze Positive'
        else:
            return 'False Negative'
    else:
        if prediction == 1:
            return 'False Positive'
        elif prediction == -1:
            return 'Reanalyze Negative'
        else:
            return 'True Negative'

lipid_data['confusion'] = [confusion(l, p) for l, p in zip(lipid_data['label'], lipid_data['pred_label'])]

cat_colors = {'True Positive':'navy', 'Reanalyze Positive':'blue', 'False Negative':'deepskyblue',
              'False Positive':'maroon', 'Reanalyze Negative':'red', 'True Negative':'lightsalmon'}
for pair in combinations(predictor_cols, 2):
    fig, ax = plt.subplots(figsize = (6,6))
    for cat in cat_colors.keys():
        lipids = lipid_data[lipid_data['confusion'] == cat]
        ax.scatter(lipids[pair[0]], lipids[pair[1]],
                   s = 1, color = cat_colors[cat], marker = '.', label = cat)
    ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    if log_pred[pair[0]]:
        ax.set_xscale('log')
    elif log_pred[pair[1]]:
        ax.set_yscale('log')
    ax.set_facecolor('lightgrey')
    ax.set_ylabel(pair[1])
    ax.set_xlabel(pair[0])
    fig.savefig(f'{args.out_dir}/training_QC/{pair[0].replace("/","")}-{pair[1].replace("/","")}_categories.svg', 
                bbox_inches = 'tight')
    plt.close('all')

#individual predictors correlation with final scores
for predictor in predictor_cols:
    fig, ax = plt.subplots(figsize = (6,6))
    ax.scatter(lipid_data[lipid_data['label'] == 1][predictor],
               lipid_data[lipid_data['label'] == 1]['score'],
               s = 1, c = 'k', marker = '.', label = 'True')
    ax.scatter(lipid_data[lipid_data['label'] == 0][predictor],
               lipid_data[lipid_data['label'] == 0]['score'],
               s = 1, c = 'r', marker = '.', label = 'False')
    ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    if log_pred[predictor]:
        ax.set_xscale('log')
    ax.set_ylabel('Score')
    ax.set_xlabel(predictor)
    fig.savefig(f'{args.out_dir}/training_QC/{predictor.replace("/","")}.svg', bbox_inches = 'tight')

def TPR(tn,fp,fn,tp):
    divisor = tp + fn
    if divisor == 0:
        return 0
    else:
        return tp/divisor

def FPR(tn,fp,fn,tp):
    divisor = fp + tn
    if divisor == 0:
        return 0
    else:
        return fp/divisor

for train,data in enumerate([lipid_data[lipid_data['test_set']],lipid_data[np.logical_not(lipid_data['test_set'])]]):
    aucroc = roc_auc_score(data['label'], data['score'])
    
    tprs = [0]
    fprs = [0]
    for cut in sorted(list(set(data['score'])), reverse=True):
        calls = [yhat >= cut for yhat in data['score']]
        tn,fp,fn,tp = confusion_matrix(data['label'], calls).flatten()
        tprs.append(TPR(tn,fp,fn,tp))
        fprs.append(FPR(tn,fp,fn,tp))
    
    tprs.append(1)
    fprs.append(1)
    
    fig, ax = plt.subplots(figsize = (6,6))
    ax.plot(fprs,tprs,'-k', linewidth = 1)
    ax.plot([0,1],[0,1], '--r', linewidth = 0.5)
    y0,y1 = ax.get_ylim()
    x0,x1 = ax.get_xlim()
    ax.set_aspect(abs(x1-x0)/abs(y1-y0))
    ax.set_ylim(-.001,1.001)
    ax.set_xlim(-.001,1.001)
    ax.set_facecolor('lightgrey')
    ax.set_ylabel('True Postiive Rate')
    ax.set_xlabel('False Postiive Rate')
    ax.set_title(f'{"Train" if train else "Test"} Set ROC')
    ax.annotate(f'AUC: {"%.2f"%(aucroc)}', (0.5,0.5), ha='left', va='top')
    fig.savefig(f'{args.out_dir}/training_QC/{"train" if train else "test"}_roc.svg', bbox_inches = 'tight')
