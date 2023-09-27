"""Utilities that are required by gplearn.

Most of these functions are slightly modified versions of some key utility
functions from scikit-learn that gplearn depends upon. They reside here in
order to maintain compatibility across different versions of scikit-learn.

"""

import numbers

import numpy as np
from joblib import cpu_count


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def _get_n_jobs(n_jobs):
    """Get number of jobs for the computation.

    This function reimplements the logic of joblib to determine the actual
    number of jobs depending on the cpu count. If -1 all CPUs are used.
    If 1 is given, no parallel computing code is used at all, which is useful
    for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
    Thus for n_jobs = -2, all CPUs but one are used.

    Parameters
    ----------
    n_jobs : int
        Number of jobs stated in joblib convention.

    Returns
    -------
    n_jobs : int
        The actual number of jobs as positive integer.

    """
    if n_jobs < 0:
        return max(cpu_count() + 1 + n_jobs, 1)
    elif n_jobs == 0:
        raise ValueError('Parameter n_jobs == 0 has no meaning.')
    else:
        return n_jobs


def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    n_jobs = min(_get_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = (n_estimators // n_jobs) * np.ones(n_jobs,
                                                              dtype=int)
    n_estimators_per_job[:n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()

def GenerateDataByDataFrame(data, tgt, weight_col=[], sequence_col=[], group_col=[]):
    X = []
    Y = []
    W = []

    if sequence_col:
        data = data.sort_values(sequence_col)

    if (group_col):
        for index, rows in data.groupby(group_col):
            if sequence_col:
                tmp = rows.sort_values(sequence_col)
            tmp = np.array(tmp.drop([tgt]+sequence_col+group_col,axis=1))
            tmp = tmp[:,:,np.newaxis]
            X.append(tmp)

        for index, rows in data.groupby(group_col):
            if sequence_col:
                tmp = rows.sort_values(sequence_col)
            tmp_y = np.array(tmp[tgt]).reshape(-1,1)
            tmp_y = tmp_y[:,:,np.newaxis]
            Y.append(tmp_y)
            if weight_col:
                tmp_w = np.array(tmp[weight_col]).reshape(-1,1)
                tmp_w = tmp_w[:,:,np.newaxis]
                W.append(tmp)
    else:
        tmp = np.array(data.drop([tgt]+sequence_col+group_col,axis=1))
        tmp = tmp[:,:,np.newaxis]
        X.append(tmp)
        tmp = np.array(data[tgt]).reshape(-1,1)
        tmp = tmp[:,:,np.newaxis]
        Y.append(tmp)


    X = np.concatenate(X,axis=2)
    Y = np.concatenate(Y,axis=2)[:, 0, :]

    if (weight_col):
        W = np.concatenate(W,axis=2)[:, 0, :]
    else:
        W = np.ones((Y.shape[0], Y.shape[-1]))

    column_names = [i for i in data.columns if i not in [tgt]+sequence_col+group_col+weight_col]

    return X, Y, W, column_names


