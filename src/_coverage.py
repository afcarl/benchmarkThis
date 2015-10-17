"""
Includes various coverage statistics, namely the binomial segmentation algorithm


"""
from __future__ import division

from skbio.stats.composition import closure
from skbio.stats import subsample_counts
import numpy as np
import matplotlib.pyplot as plt
import sympy as sy
from scipy.optimize import newton
from scipy.misc import comb, logsumexp

# First calculate root function, its first derivative and second derivative
def f(x, n, phi, N):
    """
    Function to optimize to estimate coverage

    Parameters
    ----------
    x : float
        estimated proportion
    n : array_like
        sampling depths for each sample
    N : int
        number of samples
    phi : int
        number of sucesses
    """
    return np.array([(1 - x)**n[j] for  j in range(len(n))]).sum() - N + phi

def fprime(x, n, phi, N):
    """ Derivative of function to optimize """
    return np.array([-n[j]*(1 - x)**(n[j]-1) for  j in range(len(n))]).sum()

def fprime2(x, n, phi, N):
    """ 2nd derivative of function to optimize """
    return np.array([n[j]*(n[j]-1)*(1 - x)**(n[j]-2) for  j in range(len(n))]).sum()

def replicatize(sample, reps=10):
    """
    Basically does subsampling without replacement.
    Calculates which sample has the highest abundance :math:`n`
    and obtains multiple samples of size :math:`n+1`

    Parameters
    ----------
    sample: np.array, int
        A count vector of abundances

    Returns
    -------
    mat: np.array, int
        A count matrix where
        rows = replicate samples
        columns = features
    """
    sample = np.array(sample)
    n = sample.max()

    mat = np.zeros((reps, len(sample)))
    for rep in range(reps):
        mat[rep, :] = subsample_counts(sample, n + 1)
    return mat

def binomial_segmentation(sample, reps=50):
    """
    Parameters
    ----------
    sample: np.array, int
        A count vector of abundances

    Returns
    -------
    p : np.array
        Estimated absolute proportions
    """
    # First create artificial replicates

    x0 = 0
    mat = replicatize(sample, reps)
    sample_ind = (mat>0).astype(np.int)
    phi = sample_ind.sum(axis=0)
    N = reps
    n = mat.sum(axis=1)
    p = [newton(f, x0, fprime, args=(n, phi[i], N), fprime2=fprime2) for i in range(len(phi))]
    return np.array(p)


def logcomb(N, k):
    """
    Calculates log[(n choose k)]

    Parameters
    ----------
    N : int
    k : int

    Returns
    -------
    float
    """
    num = np.log(np.arange(1, N)).sum()
    denom = np.log(np.arange(1, k)).sum() + np.log(np.arange(1, N-k)).sum()
    return num - denom


def brive(N, replace_zeros=True):
    """
    The brive estimator

    Parameters
    ----------
    N : np.array, int
       Counts vector
    replace_zeros: bool
       Replaces zeros with uniform prior

    Returns
    -------
    pvals
    """
    N = N.astype(np.int)
    n = sum(N)
    pvals = np.zeros(len(N), dtype=np.float64)
    for i in range(len(N)):
        if N[i]==0 or N[i]==1: continue
        trials = [comb(t-1, N[i]-1) / (t * (comb(n, N[i])))
                  for t in range(N[i], n+1)]
        pvals[i] = (float(N[i]-1)) * sum(trials)

    if replace_zeros:
        m = sum(pvals)
        if 0 < m < 1 and (pvals==0).sum() > 0:
            pvals[pvals==0] = (1 - m) / (pvals==0).sum()
    return pvals
