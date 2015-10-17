from __future__ import division
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from composition import closure
from metrics import variation_distance
from skbio.stats import subsample_counts
from skbio.stats.composition import closure
from skbio.diversity.alpha import robbins, lladser_pe
from composition import coverage_replacement, multiplicative_replacement
from scipy.stats import power_divergence, entropy
from cogent.parse.tree import DndParser
from cogent.maths.unifrac.fast_unifrac import fast_unifrac
from cogent.maths.unifrac.fast_tree import UniFracTreeNode
import biom
import os
import itertools
from mpl_toolkits.mplot3d import Axes3D


#######################################################################
#         Distance One Urn sampled across entire simplex              #
#######################################################################
np.random.seed(0)
data_dir = "../data/tick/meshnick_tech_reps"
biom_file = "%s/373_otu_table.biom" % data_dir
meta_file = "%s/meta.txt" % data_dir

table = biom.load_table(biom_file)
mat = np.array(table._get_sparse_data().todense()).T

# Randomly sample simplex
num_dists = 10000
num_species = 1000
depths=[300, 3000, 30000]
relative_tvd = np.zeros((num_dists, len(depths)))
robbins_tvd = np.zeros((num_dists, len(depths)))
for u, depth in enumerate(depths):
    for i in range(num_dists):
        pvals = closure(-np.log(np.random.rand(num_species)))
        # pvals = closure(mat[i, :])

        samp_table = np.random.multinomial(n=depth, pvals=pvals)

        cx1 = coverage_replacement(np.atleast_2d(samp_table),
                                   uncovered_estimator=robbins)
        relative_tvd[i, u] = variation_distance(closure(samp_table),  pvals)
        robbins_tvd[i, u] = variation_distance(cx1, pvals)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for u in range(len(depths)):
    axes[u].hist(relative_tvd[:, u], 20, label='Relative', alpha=0.5, color='b')
    axes[u].hist(robbins_tvd[:, u], 20, label='Robbins', alpha=0.5, color='r')
    axes[u].set_title('Depth=%d' % depths[u])
    if u == 0:
        axes[u].set_ylabel('Counts')
    if u == 1:
        axes[u].set_xlabel('Total Variation Distance')
    axes[u].locator_params(nbins=4)
plt.legend()
fig.savefig('../results/multiple_simplicial_hists.png')


#######################################################################
#         Distance One Urn sampled across entire simplex              #
#######################################################################

# Randomly sample simplex
num_dists = 10000
num_species = 1000
depths=np.linspace(1000, 2000000, 100)
relative_tvd = np.zeros((num_dists, len(depths)))
robbins_tvd = np.zeros((num_dists, len(depths)))
for u, depth in enumerate(depths):
    for i in range(num_dists):
        pvals = closure(-np.log(np.random.rand(num_species)))
        samp_table = np.random.multinomial(n=depth, pvals=pvals)
        cx1 = coverage_replacement(np.atleast_2d(samp_table),
                                   uncovered_estimator=robbins)

        cx2 = np.apply_along_axis(brive, 1, np.atleast_2d(samp_table),
                                  replace_zeros=True)
        relative_tvd[i, u] = variation_distance(closure(samp_table),  pvals)
        robbins_tvd[i, u] = variation_distance(cx1, pvals)
        brive_tvd[i, u] = variation_distance(cx2, pvals)

fig, axes = plt.subplots()
width = 1000
depths = depths.astype(np.int)
robbins_wins = (robbins_tvd < brive_tvd).sum(axis=0)
# axes.bar(depths, robbins_wins, width, color='r', label='Robbins')
# axes.bar(depths, num_dists - robbins_wins, width, color='b',
#          bottom=robbins_wins, label='Relative')
axes.plot(depths, robbins_wins)
axes.fill_between(depths, robbins_wins, 10000,
                  where=10000>=robbins_wins,
                  facecolor='blue', interpolate=True)
axes.fill_between(depths, robbins_wins, 0,
                  where=robbins_wins>0,
                  facecolor='red', interpolate=True)

axes.set_title('Comparison of TVD vs Sampling Depth')
axes.set_ylabel('Number of distributions')
axes.set_xlabel('Sampling depth')
plt.legend(loc=3)
plt.xlim([0, 2000000])
plt.ylim([0, 10000])
fig.savefig('../results/simplical_sampling_depth.png')


#######################################################################
#       Distance ratio One Urn sampled across entire simplex          #
#######################################################################

fig, axes = plt.subplots()
width = 1000
depths = depths.astype(np.int)
ratio = robbins_tvd / brive
# ratio = ratio[:100, :]
num_dists, _ = ratio.shape
# axes.bar(depths, robbins_wins, width, color='r', label='Robbins')
# axes.bar(depths, num_dists - robbins_wins, width, color='b',
#          bottom=robbins_wins, label='Relative')
for i in range(num_dists):
    axes.plot(depths, ratio[i, :], '-b')

axes.set_title('Comparison of TVD vs Sampling Depth')
axes.set_ylabel('TVD(robbins) / TVD(brive)')
axes.set_xlabel('Sampling depth')
plt.legend(loc=3)
plt.xlim([0, 1000000])
fig.savefig('../results/simplical_brive_robbins_ratios.png')
