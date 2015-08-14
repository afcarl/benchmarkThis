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
plt.rc('legend', **{'fontsize': 12})


num_species = 10000
num_samps = 100
pdf_dict = {
    'Geometric': np.random.geometric(1/num_species, size=num_species),
    'Uniform': np.random.uniform(5000, 15000, size=num_species)
}

depths = np.linspace(2000, 20000, 10)
disp_depths = np.linspace(2000, 20000, 4)
u, v = 0, 0
fig, axes = plt.subplots(2, 2, sharey=True)
for pdf, pval in pdf_dict.items():
    mult_mean = []
    robbins_mean = []
    pvals = closure(pval)

    for depth in depths:

        samp_table = np.array([np.random.multinomial(n=depth, pvals=pvals)
                               for i in range(num_samps)])

        mrsamp_table = multiplicative_replacement(samp_table, delta=10**-8)
        rrsamp_table = coverage_replacement(samp_table,
                                            uncovered_estimator=robbins)

        # Get both mean distortion and variance
        truth = np.tile(pvals, (num_samps, 1))
        mr_msd = mean_sq_distance(mrsamp_table, truth)
        rr_msd = mean_sq_distance(rrsamp_table, truth)
        mult_mean.append(np.mean(mr_msd))
        robbins_mean.append(np.mean(rr_msd))

    axes[u][v].semilogy(depths, mult_mean,
                        '-ob', label='Multiplicative')
    axes[u][v].semilogy(depths, robbins_mean,
                        '-or', label='Robbins')
    if u == 0:
        axes[u][v].set_xticks(disp_depths)
        axes[u][v].set_xticklabels([])
        axes[u][v].legend(loc=3)
    if u == 1:
        axes[u][v].set_xlabel('Sequencing depth')
        axes[u][v].set_xticks(disp_depths)
        axes[u][v].set_xticklabels(disp_depths)

    # axes[u].set_title('%s distribution' % pdf, fontsize=14)
    # axes[u].set_ylabel('Mean Squared Aitchison Distance')
    axes[u][v].get_xaxis().get_major_formatter().labelOnlyBase = False
    u += 1

# fig.suptitle('Distortion on Zero replacement', fontsize=18)
fig.text(0.005, 0.5, 'Mean Squared Aitchison Distance',
         va='center', rotation='vertical')
# fig.tight_layout(rect=[0, 0.03, 1, 0.95])
# fig.savefig('../results/zeros/distortion_vs_depth.png')


#######################################################################
#                     Distortion vs Delta                             #
#######################################################################

np.random.seed(0)
# plt.rc('legend', **{'fontsize': 12})

num_species = 10000
num_samps = 100
pdf_dict = {
    'Geometric': np.random.geometric(1/num_species, size=num_species),
    'Uniform': np.random.uniform(5000, 15000, size=num_species)
}
u, v = 0, 1
logdeltas = np.linspace(-12, -4, 10)
deltas = 10 ** logdeltas
# fig, axes = plt.subplots(2, sharey=True)
depth = 20000
for pdf, pval in pdf_dict.items():
    mult_mean = []
    robbins_mean = []
    pvals = closure(pval)

    for delta in deltas:

        samp_table = np.array([np.random.multinomial(n=depth, pvals=pvals)
                               for i in range(num_samps)])

        mrsamp_table = multiplicative_replacement(samp_table, delta=delta)
        rrsamp_table = coverage_replacement(samp_table,
                                            uncovered_estimator=robbins)

        # Get both mean distortion and variance
        truth = np.tile(pvals, (num_samps, 1))
        mr_msd = mean_sq_distance(mrsamp_table, truth)
        rr_msd = mean_sq_distance(rrsamp_table, truth)
        mult_mean.append(np.mean(mr_msd))
        robbins_mean.append(np.mean(rr_msd))

    axes[u][v].loglog(deltas, mult_mean,
                      '-ob', label='Multiplicative')
    axes[u][v].loglog(deltas, robbins_mean,
                      '-or', label='Robbins')
    if u == 0:
        axes[u][v].set_xticklabels([])
    if u == 1:
        axes[u][v].set_xlabel(r'$\displaystyle\delta$')
    # axes[u][v].set_title('%s distribution' % pdf, fontsize=14)
    axes[u][v].get_xaxis().get_major_formatter().labelOnlyBase = False
    axR = axes[u][v].twinx()
    axR.set_yticks([])
    axR.set_ylabel('%s distribution' % pdf, fontsize=14)
    u += 1

fig.suptitle('Distortion on Zero replacement', fontsize=18)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig('../results/zeros/distortion.png')
