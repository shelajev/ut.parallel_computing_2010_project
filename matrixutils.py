# -*- coding: utf-8 -*-

"""Additional matrix utilities for working with scipy"""

import numpy as np
import scipy.sparse as sparse

def ListToMatrix(mtxs):
    """ converts list of [ (rows, matrix) ], to a matrix  """
    mtxs.sort(key = lambda x : x[0])
    mtxs = map(lambda x : x[1], mtxs)
    R = sparse.vstack(mtxs)
    return R

def UniqueList(a):
    """ returns unique elements given a list """
    return {}.fromkeys(a).keys()

def ValueColumns(a):
    """ vector that contains col numbers for each column that contains values """
    # nonzero cols
    if sparse.isspmatrix_csr(a):
        nzcols = a.indices
    else:
        nzcols = a.nonzero()[1]
    # remove duplicates and sort
    M = sorted(UniqueList(nzcols))
    return M

def SelectRows(A, rows, mask):
    """ applies colmask 'mask' to matrix 'a' limited to 'rows'
        A is a partial matrix that represents 'rows'
        mask is total matrix based
    """
    if sparse.isspmatrix_csr(A):
        # this chooses only those rows that we want
        indptr = np.zeros(A.shape[0] + 1, dtype=int)
        indices = []
        data = []
        
        lastpos=0
        lastt=0
        count = 0
        for i in mask:
            if (i >= rows[0]) and (i < rows[1]):
                t = i - rows[0]
                # set rows between to zero
                indptr[lastt+1:t+1] = count
                # find indices/data range
                indstart, indend = A.indptr[t], A.indptr[t+1]
                # copy col_indices and data
                indices.append(A.indices[indstart:indend])
                data.append(A.data[indstart:indend])
                # set end
                count += indend - indstart
                lastt = t
        indptr[lastt+1:len(indptr)] = count
        data = np.concatenate(data)
        indices = np.concatenate(indices)
        at = sparse.csr_matrix((data,indices,indptr), dtype=A.dtype, shape=A.shape)
    else:
        s = A.shape[0]
        data = np.zeros(s)
        for i in mask:
            if (i >= rows[0]) and (i < rows[1]):
                t = i - rows[0]
                data[t] = 1.0
        mtx = sparse.spdiags(data, [0], s,s)
        at = (mtx.transpose() * a).tocsr()
    return at

