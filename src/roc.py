"""
Build ROC curves using the following
- 1-R^2 for Pearson and Spearman
- phi for Lovell

Just to check the presence of an edge
"""
import numpy as np


def confusion_matrices(corr_mat, true_corr_mat, thresholds=None):
    """
    Parameters
    ----------
    corr_mat : np.array
        esimated correlation matrix
    true_corr_mat : np.array
        true correlation matrix
    threshold : np.array
        array of thresholds to determine
        if a threshold is present or not

    Returns
    -------
    TP : np.array
       True positives
    FP : np.array
       False positives
    TN : np.array
       True negatives
    FN: np.array
       False negatives
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 20)

    def confusion_matrix(mat, exp_mat, threshold):
        _mat = mat > threshold
        _TP = np.triu(np.logical_and(_mat == 1, exp_mat == 1)).sum()
        _FP = np.triu(np.logical_and(_mat == 1, exp_mat == 0)).sum()
        _TN = np.triu(np.logical_and(_mat == 0, exp_mat == 0)).sum()
        _FN = np.triu(np.logical_and(_mat == 0, exp_mat == 1)).sum()
        return _TP, _FP, _TN, _FN

    confusion_wrap = lambda x: confusion_matrix(corr_mat, true_corr_mat, x)
    TP, FP, TN, FN = zip(*map(confusion_wrap, thresholds))
    return TP, FP, TN, FN
