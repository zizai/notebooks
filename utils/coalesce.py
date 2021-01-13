import numpy as np
import scipy.sparse as sps


def coalesce(index, value, m, n):
    """coalesce values in a sparse matrix"""
    row, col = index
    if value is None:
        value = np.ones(row.size)
    out = sps.coo_matrix((value, (row, col)), shape=(m, n)).todok()
    return out
