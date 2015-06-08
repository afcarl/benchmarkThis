"""
This is where all of benchmarking metrics will be stored

"""
from __future__ import division
import pandas as pd
import numpy as np
from roc import confusion_matrices
from composition import distance


def subcompositional_stress(X, Y):
    """
    A measure of subcompositional coherence

    Parameters
    ----------
    X : np.ndarray
        2-D array of compositions
        rows = compositions
        columns = components
    Y : np.ndarray
        2-D array of compositions
        rows = compositions
        columns = components


    Returns
    -------
    float : measure of stress
    """
    pass


def mean_sq_distance(X, Y):
    """
    A measure of subcompositional coherence.

        Parameters
        ----------
        X : np.ndarray
            2-D array of compositions
            rows = compositions
            columns = components
        Y : np.ndarray
            2-D array of compositions
            rows = compositions
            columns = components

    Returns
    -------
    float : mean squared Aitchison distance

    Notes
    -----
    X and Y must have the same number of rows and columns
    """
    X, Y = np.atleast_2d(X), np.atleast_2d(Y)
    r1, c1 = X.shape
    r2, c2 = Y.shape
    if not (r1 == r2 and c1 == c2):
        raise ValueError("The dimensions of X and Y aren't consistent")
    return np.mean(distance(X, Y)**2)


def variation_distance(p1, p2):
    """
    Calculates the total variation distance between any two compositions

    Parameters
    ----------
    p1 : np.array
       composition vector
    p2 : np.array
       composition vector

    Returns
    -------
    float :
       Total variation distance of probability

    References
    ----------
    .. http://en.wikipedia.org/wiki/
       Total_variation_distance_of_probability_measures
    """
    return 0.5*abs((p1-p2)).sum()


def confusion_evaluate(corr_mat, meas_corrs, method_names):
    """
    Determine sensitivity, specificity and precision
    for each of the estimated correlation matrices
    over a set of thresholds.  This information
    is used to generate ROC curves and precision
    recall curves

    Parameters
    ----------
    corr_mat : np.array
       true correlation matrix
    meas_corrs : list, np.array
       list of measured correlation matrices
    method_names : list, str
       list of strings for the methods

    Returns
    -------
    metric_df : pd.DataFrame
        DataFrame containing sensitivity,
        specificity and precision information
        for each of the evaluated methods
    """

    assert len(meas_corrs) == len(method_names), "Each matrix needs a name"

    thresholds = np.linspace(0, 1, 100)
    metric_df = pd.DataFrame()
    for i in range(len(meas_corrs)):
        corr = meas_corrs[i]
        TP, FP, TN, FN = confusion_matrices(corr,
                                            corr_mat,
                                            thresholds=thresholds)
        TP, FP, TN, FN = map(np.array, [TP, FP, TN, FN])

        sens, spec, prec = TP/(TP+FN), TN/(FP+TN), TP/(TP+FP)

        metric_df[method_names[i]] = pd.Series({'Sensitivity': sens,
                                                'Specificity': spec,
                                                'Precision': prec})

    return metric_df
