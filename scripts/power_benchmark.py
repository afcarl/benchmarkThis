"""
Tests power analysis on UniFrac
"""
from __future__ import division

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from composition import closure
from metrics import effect_size
from skbio.stats import subsample_counts
from skbio.diversity.alpha import robbins
from composition import coverage_replacement
from ancom import ancom
import scipy.stats as sps
import os
from generators.diff_otu import diff_multiple_otu

########################################################
# Single OTU differential benchmark
#
# Alpha (fdr) versus effect size (for 1 OTU)
# Beta (power) versus effect size (for 1 OTU)
# TODO: May want to expand this to more species
########################################################
font = {'family': 'normal',
        'weight': 'normal',
        'size': 16}

matplotlib.rc('font', **font)

np.random.seed(0)

C = 200
num_samps = 1000
num_species = 100
alpha = 0.05/num_species
beta = 0.8
diffs = np.linspace(0, .25, 25)

# Relative proportions
rel_fdr = np.zeros((len(diffs)))  # False positive rate
rel_fnr = np.zeros((len(diffs)))  # False negative rate
# Coverage corrected proportions
cov_fdr = np.zeros((len(diffs)))  # False positive rate
cov_fnr = np.zeros((len(diffs)))  # False negative rate
effect_sizes = np.zeros((len(diffs)))  # Effect size

metric_dict = {'Mann_Whitney': (sps.mannwhitneyu, '-ob'),
               't_test': (sps.ttest_ind, '-og'),
               'ancom': (ancom, '-or')}

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

for metric, metric_utils in metric_dict.items():
    metric_func, metric_color = metric_utils
    np.random.seed(0)
    # Sample OTU table plus a metadata vector for before/after spiking OTU1
    samp_table = np.zeros((num_samps, num_species), dtype=np.int64)
    cats = np.zeros((num_samps))

    for j, diff in enumerate(diffs):

        x = np.zeros((num_species))
        y = np.zeros((num_species))
        x[0] = 0.05  # fixed proportion of first species
        x[1:] = (1 - x[0])/(num_species - 1)
        y[0] = x[0] + diff
        y[1:] = (1 - y[0])/(num_species - 1)
        rel_pvals = np.zeros((num_samps))
        cov_pvals = np.zeros((num_samps))
        sig_species = np.zeros(((num_samps)//2, 2))

        for i in range(0, num_samps, 2):
            samp_table[i, :] = np.random.multinomial(n=C, pvals=x)
            samp_table[i+1, :] = np.random.multinomial(n=C, pvals=y)
            cats[i] = 0
            cats[i+1] = 1
            sig_species[i//2] = samp_table[i:i+2, 0]
        cov_table = coverage_replacement(samp_table,
                                         uncovered_estimator=robbins)

        if metric != 'ancom':
            fun = lambda x: metric_func(x[cats == 0], x[cats == 1])
            _, cov_pvals = np.apply_along_axis(fun, 0, cov_table)
        else:
            _, cov_pvals = metric_func(cov_table, cats)

        # Calculate effect size for first species
        effect_sizes[j] = effect_size(sig_species[:, 1],
                                      sig_species[:, 0])

        # Calculate fdr and power
        cov_detect = cov_pvals <= alpha
        cov_miss = cov_pvals > alpha
        cov_fdr[j] = (cov_detect[1:]).sum() / (cov_detect).sum()
        cov_fnr[j] = (cov_miss[0]).sum() / (cov_miss).sum()

    ax1.plot(np.ravel(effect_sizes),
             np.ravel(cov_fdr),
             metric_color,
             label=metric)

    ax2.plot(np.ravel(effect_sizes),
             np.ravel(cov_fnr),
             metric_color,
             label=metric)

res_dir = '../results/power'
fname = '%s/fdr_all.png' % (res_dir)
ax1.legend(loc=4)
ax1.set_title('Type I Error')
ax1.set_xlabel('Effect Size')
ax1.set_ylabel('False Discovery Rate')
fig1.savefig(fname)

res_dir = '../results/power'
fname = '%s/fnr_all.png' % (res_dir)
ax2.legend(loc=4)
ax2.set_title('Type II Error')
ax2.set_xlabel('Effect Size')
ax2.set_ylabel('False Negative Rate')
fig2.savefig(fname)

# Example plots
fname = '%s/proportion_bar.png' % (res_dir)
N = 10
ind = np.arange(N)  # the x locations for the groups
width = 0.4         # the width of the bars
fig3, ax3 = plt.subplots(2)
x = np.array([10] + [10]*(N-1))
y = np.array([20] + [10]*(N-1))
ax3[0].bar(ind, x, width, color='r', label='Time point 1')
ax3[0].bar(ind+width, y, width, color='b', label='Time point 2')
ax3[0].set_xticks([])
ax3[0].set_title('Species 1 doubles')
ax3[0].set_ylabel('Abundances')
ax3[0].legend()
# x = np.array([15] + [10]*(N-1))
# y = np.array([15] + [5]*(N-1))
# ax3[1].bar(ind, x, width, color='r')
# ax3[1].bar(ind+width, y, width, color='b')
# ax3[1].set_xticks([])
# ax3[1].set_title('Every species halves, except species 1')
# ax3[1].set_ylabel('Abundances')

x = closure(np.array([10] + [10]*(N-1)))
y = closure(np.array([20] + [10]*(N-1)))
ax3[1].bar(ind, x, width, color='r')
ax3[1].bar(ind+width, y, width, color='b')
ax3[1].set_title('Proportions for both scenarios')
ax3[1].set_ylabel('Proportions')
ax3[1].set_xlabel('Species')
plt.xticks(ind+width, map(str, range(1, 11)))
fig3.savefig(fname)


# Logratio difference plot
fname = '%s/logratio_diff.png' % (res_dir)
N = 10
ind = np.arange(N)  # the x locations for the groups
width = 0.4         # the width of the bars
fig, ax = plt.subplots()
x = np.array([10] + [10]*(N-1))
y = np.array([20] + [10]*(N-1))
lr_diff = np.zeros((10, 10))
for i in range(len(x)):
    for j in range(len(y)):
        lr_diff[i, j] = np.log(x[i]/x[j]) - np.log(y[i]/y[j])

im = ax.pcolor(lr_diff, cmap=plt.cm.Blues, edgecolor='black')

labels = np.arange(1, 11)
fig.colorbar(im)
for axis in [ax.xaxis, ax.yaxis]:
    axis.set(ticks=np.arange(0.5, len(labels)), ticklabels=labels)
ax.invert_yaxis()
ax.xaxis.tick_top()

fig.savefig(fname)

########################################################
# Proportion of Differientially Abundant OTUs
#
# Alpha (fdr) versus proportion of differientially abundant OTUs
# (i.e. OTU1 = 2 * OTU2)
#
# Beta (power) versus proportion of differientially abundant OTUs
# (i.e. OTU1 = 2 * OTU2)
########################################################

font = {'family': 'normal',
        'weight': 'normal',
        'size': 16}

matplotlib.rc('font', **font)

np.random.seed(0)

num_samps = 1000
num_species = 100
alpha = 0.05/num_species
beta = 0.8
diffs = np.arange(1, 25)
depth = 200
# Relative proportions
one_fdr = np.zeros((len(diffs)))       # False positive rate
one_fnr = np.zeros((len(diffs)))       # False negative rate
# Coverage corrected proportions
cov_fdr = np.zeros((len(diffs)))       # False positive rate
cov_fnr = np.zeros((len(diffs)))       # False negative rate
cats = np.ones(num_samps)
cats[num_samps//2:] = 0

metric_dict = {
    'Mann_Whitney': (sps.mannwhitneyu, 'ob'),
    't_test': (sps.ttest_ind, 'og'),
    'ancom': (ancom, 'or')}

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

for metric, metric_utils in metric_dict.items():
    metric_func, metric_color = metric_utils
    for j, diff in enumerate(diffs):
        # Using same parameters from ANCOM
        samp_table = diff_multiple_otu(num_species, diff,
                                       num_samps, depth, 2,
                                       high_low=False)
        one_table = samp_table + 1
        cov_table = coverage_replacement(samp_table,
                                         uncovered_estimator=robbins)

        if metric != 'ancom':
            fun = lambda x: metric_func(x[cats == 0], x[cats == 1])
            _, cov_pvals = np.apply_along_axis(fun, 0, cov_table)
            _, one_pvals = np.apply_along_axis(fun, 0, one_table)
        else:
            _, cov_pvals = metric_func(cov_table, cats)
            _, one_pvals = metric_func(one_table, cats)

        cov_detect = cov_pvals <= alpha
        cov_miss = cov_pvals > alpha
        cov_fdr[j] = (cov_detect[diff:]).sum() / (cov_detect).sum()
        cov_fnr[j] = (cov_miss[:diff]).sum() / (cov_miss).sum()

        one_detect = one_pvals <= alpha
        one_miss = one_pvals > alpha
        one_fdr[j] = (one_detect[diff:]).sum() / (one_detect).sum()
        one_fnr[j] = (one_miss[:diff]).sum() / (one_miss).sum()

    if metric == 'ancom':
        ax1.plot(diffs/num_species,
                 np.ravel(one_fdr),
                 ':'+metric_color)
    ax1.plot(diffs/num_species,
             np.ravel(cov_fdr),
             '-'+metric_color,
             label=metric)
    if metric == 'ancom':
        ax2.plot(diffs/num_species,
                 np.ravel(one_fnr),
                 ':'+metric_color)
    ax2.plot(diffs/num_species,
             np.ravel(cov_fnr),
             '-'+metric_color,
             label=metric)
res_dir = '../results/power'
fname = '%s/fdr_proportion_diff.png' % (res_dir)
# ax1.legend(loc=4)
ax1.set_title('Type I Error')
ax1.set_xlabel('Proportion of Differiential Taxa')
ax1.set_ylabel('False Discovery Rate')
fig1.savefig(fname)

res_dir = '../results/power'
fname = '%s/fnr_proportion_diff.png' % (res_dir)
# ax2.legend(loc=2)
ax2.set_title('Type II Error')
ax2.set_xlabel('Proportion of Differiential Taxa')
ax2.set_ylabel('False Negative Rate')
fig2.savefig(fname)

########################################################
# Power of binomial test vs corrected binomial test
#
# Alpha (fdr) versus percent rarefaction
# Beta (power) versus percent rarefaction
########################################################
