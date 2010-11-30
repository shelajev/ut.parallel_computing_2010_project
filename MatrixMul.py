'''
Created on 29.11.2010

@author: Sajjad Rizvi
'''

from mpi4py import MPI
import numpy as np

def Multiply(A,x):
    
    if (A.shape[1] != x.shape[0]):
        print "Matrices cannot be multiplied. A.columns != x.length"
        print "A.columns = " + str(A.shape[1])
        print "x.length = " + str(x.shape[0])
        return
    
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    buffer_size = 1

    columns = A.shape[1]
    rows = A.shape[0]
    
    finish_tag = 1
    master = 0
    
    statas = MPI.Status()
    
    if rank == master:
        
        result = np.empty(rows, np.float64(0.0))
        
        for k in range(rows):
            
            subResult = np.empty(1, np.float64)
            index = 0
            last = 0
            
            while (index < columns):
                sent = 0
                
                for i in range(1, size):
                    
                    if index >= columns:
                        break
                    elif index+buffer_size > columns:
                        last = columns
                    else:
                        last = index + buffer_size
                    
                    subA = A[k, index:last]
                    subX = x[index:last]
                    
                    index = index + buffer_size
                    
                    comm.Send(subA, dest=i)
                    comm.Send(subX, dest=i)
                    
                    sent = sent + 1
                    
                for i in range(sent):
                    comm.Recv(subResult, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=statas)
                    result[k] = result[k] + subResult
        
        for i in range(1, size):
            comm.Send(np.float64(0.0), tag=finish_tag, dest=i)
            
        print "A = " + str(A)
        print "x = " + str(x)
        print "Ax = " + str(result)
        
        return result
    else:
        cont = True
        A = np.empty(buffer_size, dtype=np.float64)
        x = np.empty(buffer_size, dtype=np.float64)
        
        while(cont):
            result = np.float64(0.0) 
            comm.Recv(A, source=0, tag=MPI.ANY_TAG, status=statas)
            received = statas.Get_count(datatype=MPI.DOUBLE)

            if statas.Get_tag() == finish_tag:
                cont = False
                break
            
            comm.Recv(x, source=master, tag=MPI.ANY_TAG, status=statas)
            
            for i in range(received):
                result = result + A[i] * x[i]
            
            comm.Send(result, dest=master)
        return
    pass


if __name__ == '__main__':
    A = np.ones((4,10), dtype=np.float64)
    x = np.arange(1,11, dtype=np.float64)
    
    Multiply(A,x)


