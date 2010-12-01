"""
Classes and methods for sparse matrices.
"""

#import common
import numpy as np
#import fnum
from mpi4py import MPI

OPTIMIZE_AX=False

class SparseMatrix:
    """Sparse matrix in triple storage format.

    @ivar m: number of matrix rows
    @ivar n: number of matrix columns
    @ivar nnz: number of matrix non-zero elements
    @ivar irows: array of L{nnz} row indices
    @type irows: C{numpy.ndarray(dtype=int)}
    @ivar icols: array of L{nnz} column indices
    @type icols: C{numpy.ndarray(dtype=int)}
    @ivar vals: array of L{nnz} non-zero elements' values
    @type vals: C{numpy.ndarray(dtype=float)}
    """

    def __init__(self, m, n, nnz):
        """Sparse matrix
        @arg m: number of rows
        @arg n: number of columns
        @arg nnz: number of non-zero elements"""

        self.m = m
        self.n = n
        self.nnz = nnz
        self.irows = np.zeros(nnz, dtype=int)
        self.icols = np.zeros(nnz, dtype=int)
        self.vals = np.zeros(nnz, dtype=float)

    def __str__(self):
        "String representation of the matrix."
        vs = []
        for i in xrange(self.nnz):
            vs.append("%d %d %.15f" % (self.irows[i],self.icols[i],self.vals[i]) )
        vs_str = "\n".join(vs)
        return "%d x %d nnz=%d\n%s" % (self.m,self.n,self.nnz, vs_str)

    def __mul__(self, vec):
        """Matrix vector multiplication.

        @arg vec: NumPy vector

        @todo: task 1: implement matrix-vector multiplication
        @todo: task 3: use matrix-vector multiplication in Fortran if L{OPTIMIZE_AX} is True
        """
        assert self.n == vec.size, "%d!=%d" % (self.n, vec.size)

        vec = vec.reshape(vec.size)

        vals = vec[self.icols]*self.vals

        if OPTIMIZE_AX:
            res = fnum.nested.sum(vals, self.irows+1, vec.size)
        else:
            res = np.zeros(self.m)
            for k in xrange(self.nnz):
                res[self.irows[k]] += vals[k]

        return res.reshape(res.size,1)


def as_sparse_matrix(matrix):
    "Create sparse matrix from NumPy matrix."
    irows, icols = matrix.nonzero()
    vals = matrix[irows, icols]

    result = SparseMatrix(matrix.shape[0], matrix.shape[1], irows.size)
    result.irows[:] = irows
    result.icols[:] = icols
    result.vals[:] = vals
    return result

