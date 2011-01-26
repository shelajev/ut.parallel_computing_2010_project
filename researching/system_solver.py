'''
Created on 08.11.2010

@author: shelajev
'''
from scipy.sparse.linalg import bicgstab as scipy_bicgstab

class DistributedSystemOfEquationSolver:

    def __init__(self):
      pass
    
    '''
    Currently uses scipy bicgstab, this passes strangly defined test_solver:
    return vector x such that Ax = b.
    Oleg.
    '''
    def solve(self, A, b, tol=1e-5, maxiter=None):
      return scipy_bicgstab(A, b, tol=tol, maxiter=maxiter)[0]