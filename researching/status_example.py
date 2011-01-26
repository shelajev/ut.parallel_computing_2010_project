from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
   data = {'a': 7, 'b': 3.14}
   comm.send(data, dest=1, tag=11)
elif rank == 1:
   s = MPI.Status()
   data = comm.recv(source=0, tag=11, status=s)
   print 'Data: %s' % data
   print 'Tag: %2d' % s.Get_tag()

