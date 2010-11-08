'''
Created on 08.11.2010

@author: shelajev
'''
from numpy.linalg import solve as accurate_solve, norm
from numpy.random import rand, seed as seed_random
from scipy.sparse import lil_matrix

from system_solver import DistributedSystemOfEquationSolver
import unittest

'''tests if our implementation is not very far from real solution on random matrixes'''
class SolverTest(unittest.TestCase, ):
    n = 20
    coeff = 0.4
    solver = DistributedSystemOfEquationSolver()
    
    def run_bicgstab(self, seed):
      seed_random(seed)
      A = lil_matrix((1000, 1000))
      A[0, :100] = rand(100)
      A[1, 100:200] = A[0, :100]
      A.setdiag(rand(1000))
      A = A.tocsr()
      b = rand(1000)
      x = self.solver.solve(A, b, maxiter=100000, tol=0.00001)
      x_ = accurate_solve(A.todense(), b)
      return norm(x - x_)
      
    def testName(self):
      print "testing on random matrixes, n =", self.n 
      err = 0.0
      for i in range(self.n):
        cur = self.run_bicgstab(seed=i)
        print "i =", i, "err = ", cur
        err += cur
      print "total error =", err
      self.assertTrue(err < self.n * self.coeff);

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
