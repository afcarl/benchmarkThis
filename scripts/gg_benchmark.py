"""
Performs a correlation benchmark on the Global Gut project
"""

from __future__ import division
import numpy as np
import pandas as pd
from biom import load_table
from biom import Table
from composition import zhengr

import matplotlib.pyplot as plt

meta_file = "../data/gg/GG_100nt.txt"
biom_file = "../data/gg/GG_100nt.biom"

table = load_table(biom_file)
meta_map = pd.read_table(meta_file, index_col=0)

mat = np.array(table._get_sparse_data().todense()).T
cmat = coverage_replacement(mat, uncovered_estimator=robbins)

lovell_mat = get_corr_matrix(mat, zhengr)
rlovell_mat = get_corr_matrix(mat, zhengr)
pearson_mat = get_corr_matrix(mat.T, np.corrcoef)
