import numpy as np
from mpi4py import MPI
from sparse import *

# Distributed sparse matrix
class ParSparseMatrix(SparseMatrix):
    def __init__(self, matrix, comm):
        self.matrix = as_sparse_matrix(matrix)
        self.m = self.matrix.m
        self.n = self.matrix.n

        self.comm = comm
        self.procs = comm.Get_size()
        self.rank = comm.Get_rank()

    def __str__(self):
        return 'Rank %d:\n%s' % (self.rank, self.matrix)

    def __mul__(self, vec):

        # Master: distribute the whole vector to every process:
        if self.rank == 0:
            assert self.n == vec.size
            for p in np.arange(1,self.procs):
                self.comm.send(vec, dest=p, tag=102)
            
            result = self.matrix*vec

            # receive all result parts:
            for p in np.arange(1,self.procs):
                part = self.comm.recv(source=p, tag=103)
                result = np.concatenate((result,part))

        else:
            vec = self.comm.recv(source=0, tag=102) # receive our part of vector
            result = self.matrix*vec
            self.comm.send(result, dest=0, tag=103) # send our part of result

            result = np.zeros((self.n,1)) # dummy return value for slaves

        return result
            
            
# Convert normal numpy matrix to distributed sparse matrix
def as_par_sparse_matrix(matrix, comm):
    procs = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        m,n = matrix.shape
        prows = m/procs # number of rows to send each processor

        # Distribute the matrix by blocks of rows:
        for p in np.arange(1,procs):
            comm.send(matrix[p*prows:(p+1)*prows,:], dest=p, tag=101)

        # For master, use the first block:
        matrix = matrix[0:prows,:]
    else:
        matrix = comm.recv(source=0, tag=101)

    result = ParSparseMatrix(matrix, comm)
    return result

# Parallel dot product
def dot(a, b, comm):    
    procs = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        assert a.size == b.size
        n = a.size
        prows = n/procs # number of elements for each process

        # Reshape vectors:
        a = a.reshape(n)
        b = b.reshape(n)

        # Distribute:
        for p in np.arange(1, procs):
            comm.send(a[p*prows:(p+1)*prows], dest=p, tag=105)
            comm.send(b[p*prows:(p+1)*prows], dest=p, tag=106)

        # Use the local part on root:
        a = a[0:prows]
        b = b[0:prows]

    else:
        # Receive necessary parts from master:
        a = comm.recv(source=0, tag=105)
        b = comm.recv(source=0, tag=106)

    vec = comm.gather(np.dot(a,b), root=0)

    if rank == 0:
        return np.sum(vec)
    return 0

# Find error value and broadcast it
# Used as a synchronization point between master and slaves
def error(r, comm):
    procs = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        # calculate error:
        #e = np.sqrt(np.sum(np.power(r,2))) # slow!
        e = np.sqrt(dot(r,r,comm))
    else:
        e = dot(0,0,comm) # dummy

    # broadcast the error value:
    e = comm.bcast(e, root=0)
    return e

# Distributed (n,1)-vector
class ParVector:
    def __init__(self, vec, comm):
        self.vec = vec
        self.n = vec.size

        self.comm = comm
        self.procs = comm.Get_size()
        self.rank = comm.Get_rank()
    
    def __str__(self):
        return 'Rank %d:\n%s' % (self.rank, self.vec)

def as_par_vector(vec, comm):
    procs = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        n = vec.shape[0]
        prows = n/procs # number of elements to send each slave

        # Distribute vector:
        for p in np.arange(1,procs):
            comm.send(vec[p*prows:(p+1)*prows,:], dest=p, tag=104)

        # Master uses its own part:
        vec = vec[0:prows,:]
    else:
        vec = comm.recv(source=0, tag=104)

    result = ParVector(vec, comm)
    return result

