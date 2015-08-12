"""
This is where all of benchmarking metrics will be stored

"""
from __future__ import division
import pandas as pd
import numpy as np
from roc import confusion_matrices
from composition import distance, closure


from cogent.maths.unifrac.fast_unifrac import fast_unifrac
from cogent.maths.unifrac.fast_tree import UniFracTreeNode
from skbio import TreeNode
from six import StringIO


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
    p1, p2 = closure(p1), closure(p2)
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


def unifrac_distance_matrix(table, sample_ids, otu_ids, tree):
    """
    Parameters
    ----------
    table : np.array
       Contingency table
       samples = rows
       observations = columns
    sample_ids : list, str
       List of sample ids
    otu_ids : list, str
       List of otu ids
    tree : str
       newick tree

    Returns
    -------
    np.array :
       Unifrac distance matrix
    """
    df = pd.DataFrame(table, index=sample_ids, columns=otu_ids)
    env = df.to_dict()
    res = fast_unifrac(tree, env, weighted=True)
    dist_mat = pd.DataFrame(res['distance_matrix'][0],
                            index=res['distance_matrix'][1],
                            columns=res['distance_matrix'][1])
    return dist_mat


def unifrac(p1, p2, sample_ids, otu_ids, tree):
    """
    Creates UniFrac distance between two urns

    Parameters
    ----------
    p1 : np.array
      Urn 1
    p2 : np.array
      Urn 2

    Returns
    -------
    np.array :
       Unifrac distance matrix
    """
    df = pd.DataFrame([p1, p2], index=sample_ids, columns=otu_ids)
    env = df.to_dict()
    res = fast_unifrac(tree, env, weighted=True)
    dist_mat = pd.DataFrame(res['distance_matrix'][0],
                            index=res['distance_matrix'][1],
                            columns=res['distance_matrix'][1])
    return dist_mat.ix[1, 0]


def unifrac_upgma(table, sample_ids, otu_ids, tree):
    """
    Parameters
    ----------
    table : np.array
       Contingency table
       samples = rows
       observations = columns
    sample_ids : list, str
       List of sample ids
    otu_ids : list, str
       List of otu ids
    tree : str
       newick tree

    Returns
    -------
    skbio.TreeNode :
       Tree representation of clustering
    """
    df = pd.DataFrame(mat, index=sample_ids, columns=otu_ids)
    env = df.to_dict()
    res = fast_unifrac(tree, env, weighted=True, modes=['cluster_envs'])
    return TreeNode.read(StringIO(str(res['cluster_envs'])))


def match_pre_post(dist_mat, num_samps):
    """
    Gets matched distances (i.e Pre 1 -> Post 1)
    Parameters
    ----------
    dist_mat : np.array
        Distance matrix

    Returns
    -------
    np.array :
        List of distances
    """
    pre = map(lambda x: "Pre.%d" % x, range(num_samps))
    post = map(lambda x: "Post.%d" % x, range(num_samps))
    return np.array([dist_mat.loc[pre[i], post[i]] for i in range(num_samps)])


def effect_size(x,y):
    """
    Calculates the effect size between vectors x and y
    using Cohen's effect size calculation

    Parameters
    ----------
    x: np.array
        Vector
    y: np.array
        Vector

    Returns
    -------
    effect_size : float
    """
    n1, n2 = len(x), len(y)
    s1, s2 = x.std(), y.std()
    s = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1 + n2 - 2))
    return (x.mean() - y.mean()) / s
