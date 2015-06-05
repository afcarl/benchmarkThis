"""
Benchmarks for correlations
"""

from scipy.stats import pearsonr, spearmanr
from roc import confusion_matrices
from table_factories.tableFactory_6315 import (getParams, init_data,
                                               build_contingency_table,
                                               build_correlation_matrix)
from correlation import zhengr, get_corr_matrix
import numpy as np

####################################################
# First look at the ecological linear tables
####################################################


pearson = lambda x, y: (pearsonr(x, y)[0])**2
spearman = lambda x, y: (spearmanr(x, y)[0])**2
zheng = lambda x, y: (zhengr(x, y))**2

params = getParams(sts=[1, 5],
                   interactions=['amensal',
                                 'commensal',
                                 'mutual',
                                 'parasite',
                                 'competition'])
params = init_data(params)
corr_mat = build_correlation_matrix(params)
table = build_contingency_table(params)

pearson_corr_mat = get_corr_matrix(table, pearson)
spearman_corr_mat = get_corr_matrix(table, spearman)
zheng_corr_mat = get_corr_matrix(table, zheng)

pTP, pFP, pTN, pFN = confusion_matrices(pearson_corr_mat,
                                        corr_mat)
sTP, sFP, sTN, sFN = confusion_matrices(spearman_corr_mat,
                                        corr_mat)
zTP, zFP, zTN, zFN = confusion_matrices(zheng_corr_mat,
                                        corr_mat)
pTP, pFP, pTN, pFN = map(np.array, [pTP, pFP, pTN, pFN])
sTP, sFP, sTN, sFN = map(np.array, [sTP, sFP, sTN, sFN])
zTP, zFP, zTN, zFN = map(np.array, [zTP, zFP, zTN, zFN])

pSens, pSpec, pPrec = pTP/(pTP+pFN), pTN/(pFP+pTN), pTP/(pTP+pFP)
sSens, sSpec, sPrec = sTP/(sTP+sFN), sTN/(sFP+sTN), sTP/(sTP+sFP)
zSens, zSpec, zPrec = zTP/(zTP+zFN), zTN/(zFP+zTN), zTP/(zTP+zFP)
