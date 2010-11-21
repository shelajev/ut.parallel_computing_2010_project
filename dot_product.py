from mpi4py import MPI
import numpy as np

# Important! This is still incomplete and will only work if 4 processes are used
# Call: mpiexec -n 4 python dot_product.py
# Reason: it is not yet matrix dimensions independent
# As long as you stick to integers and 12x5 arrays, you will be fine.
# Updates coming soon... After that refactoring.

if __name__ == '__main__':
  comm = MPI.COMM_WORLD
  size = comm.Get_size()
  rank = comm.Get_rank()

  if rank == 0:
    #master
    a = np.arange(60, dtype=np.int32).reshape(12,5)
    b = (np.arange(60, dtype=np.int32) - 15).reshape(12,5)
    assert a.shape == b.shape
    #TODO send a.shape to slaves
    row = 0
    while row < a.shape[0]:
      for recv in range(1,size):
        comm.Send(a[row,:], dest=recv)
        comm.Send(b[row,:], dest=recv)
        row = row + 1
    #TODO send something to slaves to indicate termination - no more rows coming
    buffer = np.empty(1, dtype=np.int32)
    total = 0
    for i in range(1,size):
      comm.Recv(buffer, source=i)
      total = total + buffer[0]
    print a
    print b
    print 'Result:', total
  else:
    #slaves
    #TODO array length hardcoded (5), use a parameter sent from the master
    first = np.empty(5, dtype=np.int32)
    second = np.empty(5, dtype=np.int32)
    sumPart = 0
    #TODO add a construction to receive any number of arrays, not just 3
    for i in range(1,size):
      comm.Recv(first, source=0)
      comm.Recv(second, source=0)
      result = first * second
      arraySum = result.sum()
      sumPart = sumPart + arraySum
    comm.Send(sumPart, dest=0)

