"""
Performs a correlation benchmark on the Global Gut project
"""

from __future__ import division
import numpy as np
import pandas as pd
from biom import load_table
from biom import Table
from correlation import zhengr, get_corr_matrix
from composition import coverage_replacement, clr
from skbio.diversity.alpha import robbins
import matplotlib.pyplot as plt

meta_file = "../data/gg/GG_100nt.txt"
biom_file = "../data/gg/GG_100nt.biom"

table = load_table(biom_file)
meta_map = pd.read_table(meta_file, index_col=0)

mat = np.array(table._get_sparse_data().todense()).T
mat = mat.astype(np.int64)
cmat = coverage_replacement(mat, uncovered_estimator=robbins)

# lovell_mat = get_corr_matrix(mat, zhengr)
rlovell_mat = get_corr_matrix(clr(cmat), zhengr)
np.savetxt('../results/gg_lovell.txt', rlovell_mat)
pearson_mat = get_corr_matrix(mat.T, np.corrcoef)
np.savetxt('../results/real_data/gg/gg_pearson.txt', pearson_mat)

heatmap = plt.pcolor(data)
