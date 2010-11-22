from mpi4py import MPI
import numpy as np

def dotProduct(a,b):
  comm = MPI.COMM_WORLD
  size = comm.Get_size()
  rank = comm.Get_rank()

  if rank == 0:
    #master who distributes elements and sums up results
    goOn = True
    element = 0
    buff = np.empty(1, dtype=np.float64)
    total = np.float64(0.0)
    while (goOn):
      for recv in range(1,size):
        if element < a.shape[0]:
          comm.Send([a[element], MPI.INT], dest=recv)
          comm.Send([b[element], MPI.INT], dest=recv)
          element = element + 1
          comm.Recv([buff, MPI.INT], source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
          total = total + buff[0]
        else:
          goOn = False
          break
    for recv in range(1,size):
      comm.Send(np.float64(0.0), dest=recv, tag=1)
    print 'A= %s' % a
    print 'B= %s' % b
    print 'Dot product is: %.2f' % total
    return total

  else:
    #slaves who calculate products
    first = np.empty(1, dtype=np.float64)
    second = np.empty(1, dtype=np.float64)
    result = np.empty(1, dtype=np.float64)
    s = MPI.Status()
    while (True):
      comm.Recv([first, MPI.INT], source=0, tag=MPI.ANY_TAG, status=s)
      if s.Get_tag() == 1:
        break
      comm.Recv([second, MPI.INT], source=0, tag=MPI.ANY_TAG, status=s)
      result = first * second
      comm.Send([result, MPI.INT], dest=0)
  

if __name__ == '__main__':
  a = np.arange(10, dtype=np.float64) + 2.5
  b = np.arange(10, dtype=np.float64) - 4.4
  dotProduct(a,b)

