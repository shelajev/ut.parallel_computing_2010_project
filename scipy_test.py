'''
Created on 08.11.2010

@author: shelajev
'''


from scipy.sparse import lil_matrix
from scipy.sparse.linalg import bicgstab
from numpy.linalg import solve, norm
from numpy.random import rand

def test_bicgstab():
  ''' This doens't work yet, but it shows, that we have scipy bicgstab implementation. Oleg  '''
  A = lil_matrix((1000, 1000))
  A[0, :100] = rand(100)
  A[1, 100:200] = A[0, :100]
  A.setdiag(rand(1000))

  A = A.tocsr()
  b = rand(1000)
  x = bicgstab(A, b)
  x_ = solve(A.todense(), b)
  err = norm(x-x_)
  print err < 1e-10

if __name__ == '__main__':
    test_bicgstab()
    