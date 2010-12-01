import numpy as np
import time, sys
import parallel, bicgstab
import mappedfilereader
from mpi4py import MPI

# Read in the data from a file:
def from_file(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    file.close()

    print 'lines in file:', len(lines)
    n = len(lines)/2
    #print 'n:', n

    urls = range(n) # maps IDs to URLs
    #print urls

    G = np.matrix(np.zeros(n*n,dtype=np.uint32).reshape(n,n)) # adjacency matrix

    for i in np.arange(0, len(lines), 2):
        [id,url] = lines[i].strip().split(' ', 1)
        urls[int(id)-1] = url

        children = lines[i+1].strip().split(' ')
        # fill in the adjacency matrix:
        for c in children:
            if c != '':
                G[int(c)-1,int(id)-1] = 1;

    return n, G, urls

if __name__ == '__main__':

    # MPI init:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    master = (rank == 0)

    if master:
        # read G from file and find PageRank:
        #n, G, urls = from_file('./output-biit.txt')
	
	# TODO test
	mapName = '../data/Map for crawledResults5.txt.txt' 
	mappedName = '../data/Mapped version of crawledResults5.txt.txt'

	r = MatReader(mapName, mappedName)
	n, G = r.read()

        print 'n:', n
        #print urls
        #outdeg = np.ravel(np.sum(G, axis=0)) # column sums
        #indeg = np.ravel(np.sum(G, axis=1)) # row sums
        #print outdeg
        #print indeg

        density = float(np.sum(G))/n**2
        print 'density:', density

        p = 0.85 # constant
        delta = (1-p)/n
        print 'p:', p, 'delta:', delta

        # PageRank algorithm itself:
        print 'calculating PageRank using sparse matrices:'

        colsum = np.ravel(np.sum(G, axis=0)) # column sums

        D = np.matrix(np.zeros(n*n,dtype=np.float).reshape(n,n)) # diagonal matrix
        for j in np.arange(n):
            if colsum[j] != 0:
                D[j,j] = 1.0/colsum[j]

        e = np.ones((n,1))
		
        I = np.matrix(np.zeros(n*n,dtype=np.float).reshape(n,n)) # identity matrix
        for i in np.arange(n):
            I[i,i] = 1
	
        A = I - p*G*D
        #print 'A:\n', A
    else:
        A = None
        e = None

    # create a sparse matrix:
    AS = parallel.as_par_sparse_matrix(A, comm)

    # solve the system of linear equations:
    start = time.clock()
    x = bicgstab.solve(AS,e,comm)
    total = time.clock() - start

    if master:
        x = x/np.sum(x) # scale, so that sum(x) = 1
        print 'x:\n',x # the actual PageRank vector (not sorted)
        print 'sum:', np.sum(x) # should always be 1.0
        print 'time:', total

