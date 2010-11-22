from mpi4py import MPI
import numpy as np

# Work in progress!

if __name__ == '__main__':
  comm = MPI.COMM_WORLD
  size = comm.Get_size()
  rank = comm.Get_rank()

  if rank == 0:
    #master
    a = np.arange(60, dtype=np.int32)
    b = np.arange(60, dtype=np.int32) - 15
    assert a.shape == b.shape
    element = 0
    while element < a.shape[0]:
      for recv in range(1,size):
        comm.Send(a[element], dest=recv, tag=1)
        comm.Send(b[element], dest=recv, tag=2)
        row = row + 1
    for recv in range(1,size):
      comm.Send(0, dest=recv, tag=0)
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
    first = np.int32(0)
    second = np.int32(0)
    result = np.int32(0)
    bool complete = False
    while (not complete):
      comm.Recv(first, source=0, tag=1)
      comm.Recv(second, source=0, tag=2)
      result = first * second
      comm.Send(result, dest=0)

