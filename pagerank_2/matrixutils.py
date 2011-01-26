# -*- coding: utf-8 -*-

"""Additional matrix utilities for working with scipy"""

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
    """ vector that contains col numbers for each column that contains values and
        0 otherwise"""
    # nonzero cols
    if sparse.isspmatrix_csr(a):
        nzcols = a.indices
    else:
        nzcols = a.nonzero()[1]
    # remove duplicates and sort
    M = sorted(uniqlist(nzcols))
    return M