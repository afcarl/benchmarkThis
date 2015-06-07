"""
Benchmarks for correlations
"""
from __future__ import division
from scipy.stats import spearmanr
from roc import confusion_matrices
from table_factories.tableFactory_6315 import (getParams, init_data,
                                               build_contingency_table,
                                               build_correlation_matrix)
from correlation import zhengr, get_corr_matrix
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.distributions import lognorm
from generators.ecological import *
import biom
import pandas as pd


def evaluate(corr_mat, meas_corrs, method_names):
    """
    Determine sensitivity, specificity and precision
    for each of the estimated correlation matrices
    over a set of thresholds.  This information
    is used to generate ROC curves and precision
    recall curves

    Parameters
    ----------
    corr_mat : np.array
       true correlation matrix
    meas_corrs : list, np.array
       list of measured correlation matrices
    method_names : list, str
       list of strings for the methods

    Returns
    -------
    metric_df : pd.DataFrame
        DataFrame containing sensitivity,
        specificity and precision information
        for each of the evaluated methods
    """

    assert len(meas_corrs) == len(method_names), "Each matrix needs a name"

    thresholds = np.linspace(0, 1, 100)
    metric_df = pd.DataFrame()
    for i in range(len(meas_corrs)):
        corr = meas_corrs[i]
        TP, FP, TN, FN = confusion_matrices(corr,
                                            corr_mat,
                                            thresholds=thresholds)
        TP, FP, TN, FN = map(np.array, [TP, FP, TN, FN])

        sens, spec, prec = TP/(TP+FN), TN/(FP+TN), TP/(TP+FP)

        metric_df[method_names[i]] = pd.Series({'Sensitivity': sens,
                                                'Specificity': spec,
                                                'Precision': prec})

    return metric_df


def plot_roc(metric_df, plot_styles):
    """
    Parameters
    ----------
        metric_df : pd.DataFrame
            columns : method names
            rows : sensitivity, specificity and precision

            DataFrame containing sensitivity,
            specificity and precision information
            for each of the evaluated methods
        plot_styles : list, str
            list of plotting styles for each method
            i.e. 'r' for straight red lines or 'ob'
            for blue points
    Returns
        plt.figure
    """
    roc_fig = plt.figure()
    for i, method_name in enumerate(metric_df.columns):
        spec = metric_df[method_name]['Specificity']
        sens = metric_df[method_name]['Sensitivity']
        plt.plot(1-spec, sens, plot_styles[i], label=method_name)
    plt.legend(loc=4)
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    return roc_fig


def plot_recall(metric_df, plot_styles):
    """
    Parameters
    ----------
        metric_df : pd.DataFrame
            columns : method names
            rows : sensitivity, specificity and precision

            DataFrame containing sensitivity,
            specificity and precision information
            for each of the evaluated methods
        plot_styles : list, str
            list of plotting styles for each method
            i.e. 'r' for straight red lines or 'ob'
            for blue points
    Returns
        plt.figure
    """
    recall_fig = plt.figure()
    for i, method_name in enumerate(metric_df.columns):
        sens = metric_df[method_name]['Sensitivity']
        prec = metric_df[method_name]['Precision']
        plt.plot(sens, prec, plot_styles[i], label=method_name)
    plt.legend(loc=2)
    plt.title('Precision/Recall curve')
    plt.xlabel('Recall Rate')
    plt.ylabel('Precision Rate')
    return recall_fig


###############################################
#  Main code
###############################################

# Setup
np.random.seed(0)
params = getParams(sts=[11],
                   interactions=['commensal'])

params = init_data(params, num_samps=100)
corr_mat = build_correlation_matrix(params)
table = build_contingency_table(params)

# save table
bT = biom.Table(table, range(table.shape[0]), range(table.shape[1]))
biomname = '../data/tables_6_3_2015/bioms/table_1.biom'
txtname = '../data/tables_6_3_2015/txts/table_1.txt'
open(biomname, 'w').write(bT.to_json('Jamie'))
open(txtname, 'w').write(bT.to_tsv())


zheng = lambda x, y: abs(zhengr(x, y))

#######################################################################
#                  Absolute Ecological Relations                      #
#######################################################################
pearson_corr_mat = abs(np.corrcoef(table.T))
spearman_corr_mat = abs(spearmanr(table)[0])
zheng_corr_mat = get_corr_matrix(table, zheng)
# Can insert sparcc_corr_mat right here.  Just need to
# 1. read text file of correlations via pandas DataFrame
# 2. extract matrix via the pandas as_matrix() command
# 3. take absolute value via pandas abs() command

metric_df = evaluate(corr_mat, [pearson_corr_mat,
                                spearman_corr_mat,
                                zheng_corr_mat],
                               ['Pearson', 'Spearman', 'Lovell'])

roc_fig = plot_roc(metric_df, ['-ob', '-og', '-or'])
prec_fig = plot_recall(metric_df, ['-ob', '-og', '-or'])

roc_fig.savefig('../results/simple_nonzero_eco_roc_curve.png')
prec_fig.savefig('../results/simple_nonzero_eco_pre_recall_curve.png')

#######################################################################
#                        Uniform rarefaction                          #
#######################################################################
pvals = np.apply_along_axis(lambda x: x / x.sum(), axis=1, arr=table)
samp_table = np.apply_along_axis(
    lambda p: np.random.multinomial(n=2000, pvals=p),
    axis=1, arr=pvals)

pearson_corr_mat = abs(np.corrcoef(samp_table.T))
spearman_corr_mat = abs(spearmanr(samp_table)[0])
zheng_corr_mat = get_corr_matrix(samp_table, zheng)

metric_df = evaluate(corr_mat, [pearson_corr_mat,
                                spearman_corr_mat,
                                zheng_corr_mat],
                               ['Pearson', 'Spearman', 'Lovell'])

roc_fig = plot_roc(metric_df, ['-ob', '-og', '-or'])
prec_fig = plot_recall(metric_df, ['-ob', '-og', '-or'])

roc_fig.savefig('../results/uniform_rare_eco_roc_curve.png')
prec_fig.savefig('../results/uniform_rare_eco_pre_recall_curve.png')

bT = biom.Table(samp_table,
                range(samp_table.shape[0]),
                range(samp_table.shape[1]))
biomname = '../data/tables_6_3_2015/bioms/table_2.biom'
txtname = '../data/tables_6_3_2015/txts/table_2.txt'
open(biomname, 'w').write(bT.to_json('Jamie'))
open(txtname, 'w').write(bT.to_tsv())

#######################################################################
#                        Random rarefaction                           #
#######################################################################
samp_table = np.apply_along_axis(
    lambda p: np.random.multinomial(n=np.random.geometric(1/2000)+2000,
                                    pvals=p),
    axis=1, arr=pvals)

pearson_corr_mat = abs(np.corrcoef(samp_table.T))
spearman_corr_mat = abs(spearmanr(samp_table)[0])
zheng_corr_mat = get_corr_matrix(samp_table, zheng)

metric_df = evaluate(corr_mat, [pearson_corr_mat,
                                spearman_corr_mat,
                                zheng_corr_mat],
                               ['Pearson', 'Spearman', 'Lovell'])

roc_fig = plot_roc(metric_df, ['-ob', '-og', '-or'])
prec_fig = plot_recall(metric_df, ['-ob', '-og', '-or'])

roc_fig.savefig('../results/random_rare_eco_roc_curve.png')
prec_fig.savefig('../results/random_rare_eco_pre_recall_curve.png')

bT = biom.Table(samp_table,
                range(samp_table.shape[0]),
                range(samp_table.shape[1]))
biomname = '../data/tables_6_3_2015/bioms/table_3.biom'
txtname = '../data/tables_6_3_2015/txts/table_3.txt'
open(biomname, 'w').write(bT.to_json('Jamie'))
open(txtname, 'w').write(bT.to_tsv())
