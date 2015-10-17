"""
Benchmarks for correlations
"""
from __future__ import division
from scipy.stats import spearmanr
from skbio.diversity.alpha import robbins
from table_factories.tableFactory_6315 import (getParams, init_data,
                                               build_contingency_table,
                                               build_correlation_matrix)

from correlation import zhengr, lovellr, get_corr_matrix
import numpy as np
from composition import (multiplicative_replacement,
                         coverage_replacement,
                         closure,
                         clr)
import biom
from roc import plot_roc, plot_recall
from metrics import confusion_evaluate
import matplotlib

###############################################
#  Main code
###############################################
matplotlib.rc('text', usetex=True)
# Setup
np.random.seed(0)
zheng = lambda x, y: zhengr(x, y)
res_folder = '../results/correlations/'

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

metric_df = confusion_evaluate(corr_mat, [pearson_corr_mat,
                                          spearman_corr_mat,
                                          zheng_corr_mat],
                                         ['Pearson',
                                          'Spearman',
                                          'Lovell'])

roc_fig = plot_roc(metric_df, ['-ob', '-og', '-or'])
prec_fig = plot_recall(metric_df, ['-ob', '-og', '-or'])


roc_fig.savefig('%s/simple_nonzero_eco_roc_curve.png' % res_folder)
prec_fig.savefig('%s/simple_nonzero_eco_pre_recall_curve.png' % res_folder)

#######################################################################
#                        Uniform rarefaction                          #
#######################################################################
pvals = np.apply_along_axis(lambda x: x / x.sum(), axis=1, arr=table)
samp_table = np.apply_along_axis(
    lambda p: np.random.multinomial(n=10**9, pvals=p),
    axis=1, arr=pvals)

pearson_corr_mat = abs(np.corrcoef(samp_table.T))
spearman_corr_mat = abs(spearmanr(samp_table)[0])
zheng_corr_mat = get_corr_matrix(samp_table, zhengr)

metric_df = confusion_evaluate(corr_mat, [pearson_corr_mat,
                                          spearman_corr_mat,
                                          zheng_corr_mat],
                                         ['Pearson',
                                          'Spearman',
                                          'Lovell'])

roc_fig = plot_roc(metric_df, ['-ob', '-og', '-or'])
prec_fig = plot_recall(metric_df, ['-ob', '-og', '-or'])

roc_fig.savefig('%s/uniform_rare_eco_roc_curve.png' % res_folder)
prec_fig.savefig('%s/uniform_rare_eco_pre_recall_curve.png' % res_folder)

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
font = {'family': 'normal',
        'weight': 'normal',
        'size': 16}

matplotlib.rc('font', **font)

samp_table = np.apply_along_axis(
    lambda p: np.random.multinomial(n=np.random.geometric(1/2000)+2000,
                                    pvals=p),
    axis=1, arr=pvals)

pearson_corr_mat = abs(np.corrcoef(samp_table.T))
spearman_corr_mat = abs(spearmanr(samp_table)[0])
zheng_corr_mat = get_corr_matrix(samp_table, zheng)

metric_df = confusion_evaluate(corr_mat, [pearson_corr_mat,
                                          spearman_corr_mat,
                                          zheng_corr_mat],
                                         ['Pearson',
                                          'Spearman',
                                          'Lovell'])
roc_fig = plot_roc(metric_df, ['-ob', '-og', '-or'])
prec_fig = plot_recall(metric_df, ['-ob', '-og', '-or'])

roc_fig.savefig('%s/random_rare_eco_roc_curve.png' % res_folder)
prec_fig.savefig('%s/random_rare_eco_pre_recall_curve.png' % res_folder)

bT = biom.Table(samp_table,
                range(samp_table.shape[0]),
                range(samp_table.shape[1]))
biomname = '../data/tables_6_3_2015/bioms/table_3.biom'
txtname = '../data/tables_6_3_2015/txts/table_3.txt'
open(biomname, 'w').write(bT.to_json('Jamie'))
open(txtname, 'w').write(bT.to_tsv())

#######################################################################
#                     +/- correlations benchmark                      #
#                                                                     #
# a) Tests only negative correlations                                 #
# b) Tests one negative correlation with other growing neutral factors#
#                                                                     #
# Big question: how dense does sampling need to be in order to be     #
#  sufficient to determine a correlation network?                     #
#######################################################################

# Setup
font = {'family': 'normal',
        'weight': 'normal',
        'size': 16}
matplotlib.rc('font', **font)
np.random.seed(0)
params = getParams(sts=[11],
                   interactions=[
                       'commensal',
                       'mutual',
                       'obligate_syntroph',
                       'amensal',
                       'parasite',
                       'competition'
                       ])

params = init_data(params, num_samps=5000)
corr_mat = build_correlation_matrix(params)
table = build_contingency_table(params)

# save table
bT = biom.Table(table, range(table.shape[0]), range(table.shape[1]))
biomname = '../data/tables_6_3_2015/bioms/table_4.biom'
txtname = '../data/tables_6_3_2015/txts/table_4.txt'
open(biomname, 'w').write(bT.to_json('Jamie'))
open(txtname, 'w').write(bT.to_tsv())


font = {'family': 'normal',
        'weight': 'normal',
        'size': 16}

matplotlib.rc('font', **font)
pvals = np.apply_along_axis(lambda x: x / x.sum(), axis=1, arr=table)
samp_table = np.apply_along_axis(
    lambda p: np.random.multinomial(n=np.random.uniform(5000, 20000),
                                    pvals=p),
    axis=1, arr=pvals)
rrsamp_table = coverage_replacement(samp_table,
                                    uncovered_estimator=robbins)

pearson_corr_mat = np.corrcoef(samp_table.T)
spearman_corr_mat = spearmanr(samp_table)[0]
# zheng_corr_mat = get_corr_matrix(clr(samp_table), zheng)
rrzheng_corr_mat = get_corr_matrix(clr(rrsamp_table), zhengr)
# metric_df = confusion_evaluate(corr_mat, [pearson_corr_mat,
#                                           spearman_corr_mat,
#                                           zheng_corr_mat],
#                                          ['Pearson',
#                                           'Spearman',
#                                           'Lovell'])
# roc_fig = plot_roc(metric_df, ['-ob', '-og', '-or'])
# prec_fig = plot_recall(metric_df, ['-ob', '-og', '-or'])

metric_df = confusion_evaluate(corr_mat, [pearson_corr_mat,
                                          spearman_corr_mat,
                                          # zheng_corr_mat,
                                          rrzheng_corr_mat],
                                         ['Pearson',
                                          'Spearman',
                                          # 'Lovell',
                                          'Robbins Corrected Lovell'])
font = {'family': 'normal',
        'weight': 'normal',
        'size': 16}
matplotlib.rc('font', **font)

matplotlib.rcParams.update({'font.size': 20})
roc_fig = plot_roc(metric_df, ['-ob', '-og', '-or'])
prec_fig = plot_recall(metric_df, ['-ob', '-og', '-or'], loc=None)

roc_fig.savefig('%s/all_rand_rare_eco_roc_curve.png' % res_folder)
prec_fig.savefig('%s/all_rand_rare_eco_pre_recall_curve.png' % res_folder)
