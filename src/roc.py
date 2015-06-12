"""
Build ROC curves using the following
- 1-R^2 for Pearson and Spearman
- phi for Lovell

Just to check the presence of an edge
"""
import numpy as np
import matplotlib.pyplot as plt


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


def plot_roc(metric_df, plot_styles, loc=4):
    """
    Parameters
    ----------
        metric_df : pd.DataFrame
            columns : method names
            rows : sensitivity, specificity and precision

            DataFrame containing sensitivity,
            specificity and precision information
            for each of the evaluated methods
    Returns
        plt.figure
    """
    roc_fig = plt.figure()
    for i, method_name in enumerate(metric_df.columns):
        spec = metric_df[method_name]['Specificity']
        sens = metric_df[method_name]['Sensitivity']
        plt.plot(1-spec, sens, plot_styles[i], label=method_name)
    if loc is not None:
        plt.legend(loc=loc)
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    return roc_fig


def plot_recall(metric_df, plot_styles, loc=2):
    """
    Parameters
    ----------
        metric_df : pd.DataFrame
            columns : method names
            rows : sensitivity, specificity and precision

            DataFrame containing sensitivity,
            specificity and precision information
            for each of the evaluated methods
    Returns
        plt.figure
    """
    recall_fig = plt.figure()
    for i, method_name in enumerate(metric_df.columns):
        sens = metric_df[method_name]['Sensitivity']
        prec = metric_df[method_name]['Precision']
        plt.plot(sens, prec, plot_styles[i], label=method_name)
    if loc is not None:
        plt.legend(loc=loc)
    plt.title('Precision/Recall curve')
    plt.xlabel('Recall Rate')
    plt.ylabel('Precision Rate')
    return recall_fig
