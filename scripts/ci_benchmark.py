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
from stats import coverage_correction
import matplotlib.pyplot as plt

np.random.seed(0)
num_samps = 100
num_rarefactions = 10
num_species = 100

pvals = closure(np.random.geometric(1/10000, size=num_species))
# depth = np.random.geometric(1/2000)+2000
depth = 1000
samp_table = np.random.multinomial(n=depth, pvals=pvals)
corrected_pvals = coverage_correction(samp_table, robbins)
LB_p, UB_p, LB_cover, UB_cover = multinomial_bootstrap_ci(samp_table,
                                                          robbins,
                                                          alpha=0.01,
                                                          bootstraps=10000,
                                                          random_state=0)
fig = plt.figure()
asymmetric_error = [LB_p, UB_p]

plt.errorbar(np.arange(num_species), corrected_pvals, yerr=asymmetric_error,
             fmt='o', label='Estimated Proportions')
plt.plot(np.arange(num_species), pvals, 'or', label='True Proportions')
plt.legend(loc=2)
plt.savefig('../results/ci/single_sample_ci.png')
