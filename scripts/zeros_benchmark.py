"""
A benchmark for compositional zeros
"""

from __future__ import division

from table_factories.tableFactory_6315 import (getParams, init_data,
                                               build_contingency_table,
                                               build_correlation_matrix)
from correlation import zhengr, get_corr_matrix
import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import spearmanr
import matplotlib
import biom
from roc import plot_roc, plot_recall
from metrics import confusion_evaluate, mean_sq_distance
from composition import (multiplicative_replacement,
                         coverage_replacement,
                         closure)
from skbio.diversity.alpha import robbins

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
#               Uniform rarefaction correlation                       #
#######################################################################
pvals = np.apply_along_axis(lambda x: x / x.sum(), axis=1, arr=table)
samp_table = np.apply_along_axis(
    lambda p: np.random.multinomial(n=2000, pvals=p),
    axis=1, arr=pvals)
mrsamp_table = multiplicative_replacement(samp_table)
lrsamp_table = coverage_replacement(samp_table)
rrsamp_table = coverage_replacement(samp_table,
                                    uncovered_estimator=robbins)

pearson_corr_mat = abs(np.corrcoef(samp_table.T))
spearman_corr_mat = abs(spearmanr(samp_table)[0])
zheng_corr_mat = get_corr_matrix(samp_table, zheng)
rrzheng_corr_mat = get_corr_matrix(rrsamp_table, zheng)

metric_df = confusion_evaluate(corr_mat, [pearson_corr_mat,
                                          spearman_corr_mat,
                                          zheng_corr_mat,
                                          rrzheng_corr_mat],
                                         ['Pearson',
                                          'Spearman',
                                          'Lovell',
                                          'Robbins Corrected Lovell'])

roc_fig = plot_roc(metric_df, ['-ob', '-og', '-or', '-om'])
prec_fig = plot_recall(metric_df, ['-ob', '-og', '-or', '-om'])

roc_fig.savefig('../results/zeros/uniform_rare_eco_roc_curve.png')
prec_fig.savefig('../results/zeros/uniform_rare_eco_pre_recall_curve.png')

#######################################################################
#                   Random rarefaction correlation                    #
#######################################################################
font = {'family': 'normal',
        'weight': 'normal',
        'size': 13}

matplotlib.rc('font', **font)

samp_table = np.apply_along_axis(
    lambda p: np.random.multinomial(n=np.random.geometric(1/2000)+2000,
                                    pvals=p),
    axis=1, arr=pvals)

mrsamp_table = multiplicative_replacement(samp_table)
lrsamp_table = coverage_replacement(samp_table)
rrsamp_table = coverage_replacement(samp_table,
                                    uncovered_estimator=robbins)

pearson_corr_mat = abs(np.corrcoef(samp_table.T))
spearman_corr_mat = abs(spearmanr(samp_table)[0])
zheng_corr_mat = get_corr_matrix(samp_table, zheng)
rrzheng_corr_mat = get_corr_matrix(rrsamp_table, zheng)

metric_df = confusion_evaluate(corr_mat, [pearson_corr_mat,
                                          spearman_corr_mat,
                                          zheng_corr_mat,
                                          rrzheng_corr_mat],
                                         ['Pearson',
                                          'Spearman',
                                          'Lovell',
                                          'Robbins Corrected Lovell'])

roc_fig = plot_roc(metric_df, ['-ob', '-og', '-or', '-om'])
prec_fig = plot_recall(metric_df, ['-ob', '-og', '-or', '-om'], loc=None)

roc_fig.savefig('../results/zeros/random_rare_eco_roc_curve.png')
prec_fig.savefig('../results/zeros/random_rare_eco_pre_recall_curve.png')

#######################################################################
#                   Distortion vs Rarefaction depth                   #
#######################################################################
np.random.seed(0)
pvals = closure(np.random.geometric(1/10000, size=10000))
fig = plt.figure()
mult_mean = []
robbins_mean = []

for depth in range(1000, 10000, 1000):

    samp_table = np.array([np.random.multinomial(n=depth, pvals=pvals)
                           for i in range(1000)])

    mrsamp_table = multiplicative_replacement(samp_table)
    rrsamp_table = coverage_replacement(samp_table,
                                        uncovered_estimator=robbins)

    # Get both mean distortion and variance
    truth = np.tile(pvals, (1000, 1))
    mr_msd = mean_sq_distance(mrsamp_table, truth)
    rr_msd = mean_sq_distance(rrsamp_table, truth)
    mult_mean.append(np.mean(mr_msd))
    robbins_mean.append(np.mean(rr_msd))

depths = range(1000, 10000, 1000)
plt.semilogy(depths, mult_mean,
             '-ob', label='Multiplicative')
plt.semilogy(depths, robbins_mean,
             '-or', label='Robbins')
plt.legend()
plt.title('Distortion on Zero replacement')
plt.xlabel('Sequencing depth')
plt.ylabel('Mean Squared Aitchison Distance')
fig.savefig('../results/zeros/distortion_vs_depth.png')
