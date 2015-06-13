from __future__ import division
import numpy as np
from scipy.stats import chi2
from scipy.sparse.linalg import eigsh
from composition import closure
from util import check_random_state


def robbins_variance(x):
    """
    Parameters
    ----------
    x : np.array
        Count vector

    Returns
    -------
    float :
        Upper bound for variance
    """
    return 1 / (x.sum() + 1)


def mvn_ellipsoid(mu, sigma, alpha):
    """
    Calculates the parameters for an ellipsoid assuming
    a multivariate normal distribution

    Parameters
    ----------
    mu : np.array
        Mean vector
    sigma : np.array
        Covariance matrix
    alpha : float
        signficance value

    Returns
    -------
    eigvals : np.array
        Eigenvalues of covariance matrix decomposition
    eigvecs : np.array
        Eigenvectors of covariance matrix decomposition
    half_widths : np.array
        Length of ellipsoid along each eigenvector
    """
    D = len(mu)
    eigvals, eigvecs = eigsh(sigma)
    X2 = chi2.interval(alpha, df=D)
    half_widths = np.sqrt(eigvals * X2)
    return eigvals, eigvecs, half_widths


def multinomial_bootstrap_ci(count_vec, uncovered_estimator,
                             alpha=0.05, bootstraps=1000,
                             random_state=0):
    """
    Performs a bootstrapping procedure to calculate confidence intervals

    Parameters
    ----------
    count_vec : np.array
       Count vector
    uncovered_estimator : function
       Discovery estimator (aka unobserved probability estimator)
    alpha : float
       Significance value (aka quantiles)
    bootstraps : int
       Number of bootstraps to perform
    random_state : np.random.RandomState or int
       used to generate random numbers

    Returns
    -------
    LB_p : np.array
       Lower bounds in confidence intervals for composition
    UB_p : np.array
       Upper bounds in confidence intervals for composition
    LB_cover : float
       Lower bound for discovery probability
    UB_cover : float
       Upper bound for discovery probability
    """
    random_state = check_random_state(random_state)
    N = count_vec.sum()
    p = closure(count_vec)
    boots = random_state.multinomial(N, p, size=bootstraps)
    boot_rel_p = closure(boots)
    boot_p_unobs = np.apply_along_axis(uncovered_estimator,
                                       -1, boots)
    boot_p = boot_rel_p * np.atleast_2d(1 - boot_p_unobs).T
    LB_p = np.percentile(boot_p, alpha/2 * 100, axis=0)
    UB_p = np.percentile(boot_p, (1-alpha/2) * 100, axis=0)
    LB_cover = np.percentile(boot_p_unobs, alpha/2 * 100)
    UB_cover = np.percentile(boot_p_unobs, (1-alpha/2) * 100)
    return LB_p, UB_p, LB_cover, UB_cover
