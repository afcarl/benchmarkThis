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


def evaluate(table, corr_mat):
    zheng = lambda x, y: abs(zhengr(x, y))

    pearson_corr_mat = abs(np.corrcoef(table.T))
    spearman_corr_mat = abs(spearmanr(table)[0])
    zheng_corr_mat = get_corr_matrix(table, zheng)

    pTP, pFP, pTN, pFN = confusion_matrices(pearson_corr_mat,
                                            corr_mat,
                                            thresholds=np.linspace(0, 1, 100))
    sTP, sFP, sTN, sFN = confusion_matrices(spearman_corr_mat,
                                            corr_mat,
                                            thresholds=np.linspace(0, 1, 100))
    zTP, zFP, zTN, zFN = confusion_matrices(zheng_corr_mat,
                                            corr_mat,
                                            thresholds=np.linspace(0, 1, 100))
    pTP, pFP, pTN, pFN = map(np.array, [pTP, pFP, pTN, pFN])
    sTP, sFP, sTN, sFN = map(np.array, [sTP, sFP, sTN, sFN])
    zTP, zFP, zTN, zFN = map(np.array, [zTP, zFP, zTN, zFN])

    pSens, pSpec, pPrec = pTP/(pTP+pFN), pTN/(pFP+pTN), pTP/(pTP+pFP)
    sSens, sSpec, sPrec = sTP/(sTP+sFN), sTN/(sFP+sTN), sTP/(sTP+sFP)
    zSens, zSpec, zPrec = zTP/(zTP+zFN), zTN/(zFP+zTN), zTP/(zTP+zFP)

    return (pSens, pSpec, pPrec,
            sSens, sSpec, sPrec,
            zSens, zSpec, zPrec)


def plot_curves(pSens, pSpec, pPrec,
                sSens, sSpec, sPrec,
                zSens, zSpec, zPrec):

    # Plot the simple absolute scenario
    roc_fig = plt.figure()
    plt.plot(1-pSpec, pSens, '-ob', label='pearson')
    plt.plot(1-sSpec, sSens, '-og', label='spearman')
    plt.plot(1-zSpec, zSens, '-or', label='lovell')
    plt.legend(loc=4)
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    prec_fig = plt.figure()
    plt.plot(pSens, pPrec, '-ob', label='pearson')
    plt.plot(sSens, sPrec, '-og', label='spearman')
    plt.plot(zSens, zPrec, '-or', label='lovell')
    plt.legend(loc=2)
    plt.title('Precision/Recall curve')
    plt.xlabel('Recall Rate')
    plt.ylabel('Precision Rate')

    return roc_fig, prec_fig


# Setup
params = getParams(sts=[11],
                   interactions=['commensal'])

params = init_data(params, num_samps=100)
corr_mat = build_correlation_matrix(params)
table = build_contingency_table(params)


#######################################################################
#                  Absolute Ecological Relations                      #
#######################################################################

res = evaluate(table, corr_mat)
roc_fig, prec_fig = plot_curves(*res)

roc_fig.savefig('../results/simple_nonzero_eco_roc_curve.png')
prec_fig.savefig('../results/simple_nonzero_eco_pre_recall_curve.png')

#######################################################################
#                        Uniform rarefaction                          #
#######################################################################
pvals = np.apply_along_axis(lambda x: x / x.sum(), axis=1, arr=table)
samp_table = np.apply_along_axis(
    lambda p: np.random.multinomial(n=2000, pvals=p),
    axis=1, arr=pvals)

res = evaluate(samp_table, corr_mat)
roc_fig, prec_fig = plot_curves(*res)

roc_fig.savefig('../results/uniform_rare_eco_roc_curve.png')
prec_fig.savefig('../results/uniform_rare_eco_pre_recall_curve.png')

#######################################################################
#                        Random rarefaction                           #
#######################################################################
rand_func = lambda x: np.random.geometric(1/2000)+2000
samp_table = np.apply_along_axis(
    lambda p: np.random.multinomial(n=np.random.geometric(1/2000)+2000,
                                    pvals=p),
    axis=1, arr=pvals)

res = evaluate(samp_table, corr_mat)
roc_fig, prec_fig = plot_curves(*res)

roc_fig.savefig('../results/random_rare_eco_roc_curve.png')
prec_fig.savefig('../results/random_rare_eco_pre_recall_curve.png')
