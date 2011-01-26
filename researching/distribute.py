# Idea for distributing matrix A row-by-row 
from mpi4py import MPI
import mat as m

#Returns a sample sparse m.m matrix
def testMatA():
	m = 10000
	A = m.A(m,m)



comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print "my rank is ", rank


