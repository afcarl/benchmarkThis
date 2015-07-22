import numpy as np
import scipy.stats as sps


def holm(p):
    """
    Performs Holm-Boniferroni correction for pvalues
    to account for multiple comparisons

    Parameters
    ---------
    p: numpy.array
        array of pvalues

    Returns
    =======
    numpy.array
        corrected pvalues
    """
    K = len(p)
    sort_index = -np.ones(K, dtype=np.int64)
    sorted_p = np.sort(p)
    sorted_p_adj = sorted_p*(K-np.arange(K))
    for j in range(K):
        idx = (p == sorted_p[j]) & (sort_index < 0)
        num_ties = len(sort_index[idx])
        sort_index[idx] = np.arange(j, (j+num_ties), dtype=np.int64)

    sorted_holm_p = [min([max(sorted_p_adj[:k]), 1])
                     for k in range(1, K+1)]
    holm_p = [sorted_holm_p[sort_index[k]] for k in range(K)]
    return holm_p


def _log_compare(mat, cats,
                 stat_func=sps.ttest_ind):
    """
    Calculates pairwise log ratios between all otus
    and performs a permutation tests to determine if there is a
    significant difference in OTU ratios with respect to the
    variable of interest

    otu_table: pd.DataFrame
       rows = samples
       columns = features (i.e. OTUs)
    cat: np.array, float
       Vector of categories
    stat_func: function
        statistical test to run

    Returns:
    --------
    log ratio pvalue matrix
    """
    r, c = mat.shape
    log_ratio = np.zeros((c, c))
    log_mat = np.log(mat)
    cs = np.unique(cats)
    for i in range(c-1):
        ratio =  (log_mat[:, i].T - log_mat[:, i+1:].T).T
        func = lambda x: stat_func(*[x[cats == k] for k in cs])
        m, p = np.apply_along_axis(func,
                                   axis=0,
                                   arr=ratio)
        # m, p = stat_func(ratio, cats)
        log_ratio[i, i+1:] = np.squeeze(np.array(p.T))
    return log_ratio


def ancom(otu_table, cats,
          alpha=0.05,
          multicorr=True,
          func=sps.ttest_ind):
    """
    Calculates pairwise log ratios between all otus
    and performs permutation tests to determine if there is a
    significant difference in OTU ratios with respect to the
    variable of interest

    Parameters
    ----------
    otu_table: np.array
       rows = samples
       columns = features (i.e. OTUs)
    cat: np.array, float
       Vector of categories


    Returns:
    --------
    W : np.array, float
        List of W statistics
    pvals : np.array, float
        List of pvalues
    """

    # mat = otu_table.as_matrix().transpose()
    mat = otu_table
    mat = mat.astype(np.float32)
    cats = cats.astype(np.float32)
    if np.any(mat == 0):
        raise ValueError('Cannot handle zeros in OTU table'
                         'Make sure to run a zero replacement method')
    _logratio_mat = _log_compare(mat, cats, func)
    logratio_mat = _logratio_mat + _logratio_mat.transpose()
    # np.savetxt("log_ratio.gz",logratio_mat)
    n_samp, n_otu = mat.shape
    # Multiple comparisons
    if multicorr:
        for i in range(n_otu):
            pvalues = holm(logratio_mat[i, :])
            logratio_mat[i, :] = pvalues

    W = np.zeros(n_otu)
    for i in range(n_otu):
        W[i] = (logratio_mat[i, :] < alpha).sum()
    par = n_otu-1  # cutoff

    c_start = max(W)/par
    cutoff = c_start - np.linspace(0.05, 0.25, 5)
    D = 0.02  # Some arbituary constant
    dels = np.zeros(len(cutoff))
    prop_cut = np.zeros(len(cutoff), dtype=np.float32)
    for cut in range(len(cutoff)):
        prop_cut[cut] = sum(W > par*cutoff[cut])/float(len(W))
    for i in range(len(cutoff)-1):
        dels[i] = abs(prop_cut[i]-prop_cut[i+1])

    if (dels[1] < D) and (dels[2] < D) and (dels[3] < D):
        nu = cutoff[1]
    elif (dels[1] >= D) and (dels[2] < D) and (dels[3] < D):
        nu = cutoff[2]
    elif (dels[2] >= D) and (dels[3] < D) and (dels[4] < D):
        nu = cutoff[3]
    else:
        nu = cutoff[4]
    return W, alpha*2*(W < nu*par)
