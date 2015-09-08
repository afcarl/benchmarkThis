
"""
Implements Lovell's correlation metric
"""
from __future__ import division
import numpy as np
from composition import clr


def lovellr(x, y):
    """
    Calculates proportional goodness of fit

    Parameters
    ----------
    x : array_like
    y : array_like

    Returns
    -------
    float : proportional goodness of fit
    """
    x, y = np.array(x), np.array(y)
    x_idx = (x > 0)
    y_idx = (y > 0)
    # Drop zeros
    idx = np.logical_and(x_idx, y_idx)
    m = np.vstack((x[idx], y[idx])).T
    _x, _y = clr(m).T
    return np.var(_x - _y) / np.var(_x)


def zhengr(x, y):
    """
    Calculates proportional correlation coefficient

    Parameters
    ----------
    x : array_like
    y : array_like

    Returns
    -------
    float : proportional correlation coefficient
    """
    x, y = np.array(x), np.array(y)
    x_idx = (x > 0)
    y_idx = (y > 0)
    # Drop zeros
    idx = np.logical_and(x_idx, y_idx)
    _x, _y = clr(x[idx]), clr(y[idx])
    return 2*np.cov(_x, _y)[1, 0] / (np.var(_x) + np.var(_y))


def get_corr_matrix(mat, func):
    """
    Generates a correlation matrix

    Parameters
    ----------
    mat : np.array
       Contingency table
       columns = OTUs
       rows = samples
    func : function
       correlation function

    Returns
    -------
    np.array :
       correlation matrix
    """
    r, c = mat.shape
    corr_mat = np.zeros((c, c))
    for i in range(c):
        for j in range(i):
            corr_mat[i, j] = func(mat[:, i], mat[:, j])
    corr_mat += corr_mat.T
    return corr_mat
