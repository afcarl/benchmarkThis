"""
Tests power analysis on UniFrac
"""
from __future__ import division

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from composition import closure
from metrics import variation_distance, effect_size
from skbio.stats import subsample_counts
from skbio.diversity.alpha import robbins
from composition import coverage_replacement
from ancom import ancom
import scipy.stats as sps
import biom
import os
########################################################
# Single OTU differential benchmark
#
# Alpha (fdr) versus effect size (for 1 OTU)
# Beta (power) versus effect size (for 1 OTU)
# TODO: May want to expand this to more species
########################################################
font = {'family': 'normal',
        'weight': 'normal',
        'size': 13}

matplotlib.rc('font', **font)

np.random.seed(0)

C = 200
num_samps = 1000
num_species = 100
alpha = 0.05
beta = 0.8
diffs = np.linspace(0, .25, 25)

# Relative proportions
rel_fdr = np.zeros((len(diffs)))  # False positive rate
rel_fnr = np.zeros((len(diffs)))  # False negative rate
# Coverage corrected proportions
cov_fdr = np.zeros((len(diffs)))  # False positive rate
cov_fnr = np.zeros((len(diffs)))  # False negative rate
effect_sizes = np.zeros((len(diffs)))  # Effect size

# def ancom_func(x, y):
#     n = len(x) + len(y)
#     return ancom(np.vstack((x, y)),
#                  np.arange(n) % 2)

metric_dict = {'Mann_Whitney': sps.mannwhitneyu,
               't_test': sps.ttest_ind,
               'ancom': ancom}
fig = plt.figure()
for metric, metric_func in metric_dict.items():
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
            _, rel_pvals = np.apply_along_axis(fun, 0, closure(samp_table))
            _, cov_pvals = np.apply_along_axis(fun, 0, cov_table)
        else:
            _, cov_pvals = metric_func(cov_table, cats)

        # Calculate effect size for first species
        effect_sizes[j] = effect_size(sig_species[:, 1],
                                      sig_species[:, 0])

        # Calculate fdr and power
        rel_detect = rel_pvals <= alpha
        rel_miss = rel_pvals > alpha
        cov_detect = cov_pvals <= alpha
        cov_miss = cov_pvals > alpha
        rel_fdr[j] = (rel_detect[1:]).sum() / (rel_detect).sum()
        rel_fnr[j] = (rel_miss[0]).sum() / (rel_miss).sum()
        cov_fdr[j] = (cov_detect[1:]).sum() / (cov_detect).sum()
        cov_fnr[j] = (cov_miss[0]).sum() / (cov_miss).sum()

    plt.plot(np.ravel(effect_sizes),
             np.ravel(cov_fdr), label=metric)
    # Plot fdr vs effect size
    # fig = plt.figure()
    # if metric != 'ancom':
    #     plt.plot(np.ravel(effect_sizes),
    #              np.ravel(rel_fdr),
    #              '-g',
    #              label='Relative proportions')
    # plt.plot(np.ravel(effect_sizes),
    #          np.ravel(cov_fdr),
    #          '-r',
    #          label='Corrected proportions')
    # plt.legend(loc=4)
    # plt.xlabel('Effect Size')
    # plt.ylabel('False Discovery Rate')
    # res_dir = '../results/power'
    # metric_dir = '%s/%s' % (res_dir, metric)
    # if not os.path.exists(metric_dir):
    #     os.mkdir(metric_dir)
    # fname = '%s/fdr.png' % (metric_dir)
    # fig.savefig(fname)

    # # Plot fnr vs effect size
    # fig = plt.figure()
    # if metric != 'ancom':
    #     plt.plot(effect_sizes,
    #              rel_fnr,
    #              '-g',
    #              label='Relative proportions')
    # plt.plot(effect_sizes,
    #          cov_fnr,
    #          '-r',
    #          label='Corrected proportions')
    # plt.legend(loc=4)
    # plt.xlabel('Effect Size')
    # plt.ylabel('False Negative Rate')
    # res_dir = '../results/power'
    # metric_dir = '%s/%s' % (res_dir, metric)
    # if not os.path.exists(metric_dir):
    #     os.mkdir(metric_dir)
    # fname = '%s/fnr.png' % (metric_dir)
    # fig.savefig(fname)

res_dir = '../results/power'
fname = '%s/fdr_all.png' % (res_dir)
plt.legend(loc=4)
plt.xlabel('Effect Size')
plt.ylabel('False Discovery Rate')

fig.savefig(fname)
# Alpha (fdr) versus percent rarefaction

# Alpha (fdr) versus proportion of differientially abundant OTUs
# (i.e. OTU1 = 2 * OTU2)

# Beta (power) versus percent rarefaction

# Beta (power) versus proportion of differientially abundant OTUs
# (i.e. OTU1 = 2 * OTU2)
