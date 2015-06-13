"""
Benchmark for calculating confidence intervals

Evaluates
1. Normal approximation with prior covariance matrix
2. Bootstrapping with coverage prior
3. Bootstrapping without coverage
"""
# from stats import robbins_variance, mvn_ellipsoid
from __future__ import division
import numpy as np
from stats import multinomial_bootstrap_ci
from composition import closure
from skbio.diversity.alpha import robbins

np.random.seed(0)
num_samps = 100
num_rarefactions = 10
num_species = 2000

pvals = closure(np.random.geometric(1/10000, size=num_species))
# depth = np.random.geometric(1/2000)+2000
depth = 3000
samp_table = np.random.multinomial(n=depth, pvals=pvals)

LB_p, UB_p, LB_cover, UB_cover = multinomial_bootstrap_ci(samp_table,
                                                          robbins,
                                                          alpha=0.05,
                                                          bootstraps=1000,
                                                          random_state=0)
