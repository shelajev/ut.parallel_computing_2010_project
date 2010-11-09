import numpy as np
import time, sys, operator

p = 0.85
TOLERANCE=1E-9
MAX_ITER=1000

def error(r):
    return np.sqrt(np.sum(np.power(r,2)))

# iterative pagerank calculation:
def iterative(R):

	colsum = np.ravel(np.sum(G, axis=0)) # column sums

	# form M:
	M = np.matrix(np.zeros(n*n,dtype=np.float).reshape(n,n))
	for i in np.arange(n):
		for j in np.arange(n):
			if G[i,j] == 1:
				M[i,j] = 1.0/colsum[j]
				
	#print 'M:\n', M
	#print 'det:', np.linalg.det(M)

	r = R # (initial) error vector
	e = [] # errors
	it = 0 # iteration counter

	while it < MAX_ITER and error(r) > TOLERANCE:
		it += 1
		R_old = R
		R = p*M*R + delta
		R = R/np.sum(R)
		r = R - R_old
		e.append(error(r))
		#print 'R(' + str(it+1) + '):\n', R
		#print 'error:', error(r)
		#print 'sum:', np.sum(R)
		
	return R, e

# power method		
def power_method(R):

	colsum = np.ravel(np.sum(G, axis=0)) # column sums
	
	# form B:
	B = np.matrix(np.zeros(n*n,dtype=np.float).reshape(n,n))
	for i in np.arange(n):
		for j in np.arange(n):
			if colsum[j] == 0:
				B[i,j] = 1.0/n
			else:
				B[i,j] = p*G[i,j]/colsum[j] + delta
		
	#print 'B:\n', B
	#print 'B colsum:', np.ravel(np.sum(B, axis=0))
	#print 'det:', np.linalg.det(B)
	
	r = R # (initial) error vector
	e = [] # errors
	it = 0 # iteration counter
	
	while it < MAX_ITER and error(r) > TOLERANCE:
		it += 1
		R_old = R
		R = B*R
		r = R - R_old
		e.append(error(r))
		#print 'R(' + str(it+1) + '):\n', R
		#print 'error:', error(r)
		#print 'sum:', np.sum(R)
		
	return R, e

# transfer the problem to linear system of equations:
def linear_eq(use_sparse=False):

	colsum = np.ravel(np.sum(G, axis=0)) # column sums

	D = np.matrix(np.zeros(n*n,dtype=np.float).reshape(n,n)) # diagonal matrix
	for j in np.arange(n):
		if colsum[j] != 0:
			D[j,j] = 1.0/colsum[j]
			
	e = np.ones(n).reshape(n,1)
	#zT = np.empty(n)
	#for j in np.arange(n):
	#	if colsum[j] == 0:
	#		zT[j] = 1.0/n
	#	else:
	#		zT[j] = delta
			
	I = np.matrix(np.zeros(n*n,dtype=np.float).reshape(n,n)) # identity matrix
	for i in np.arange(n):
		I[i,i] = 1
		
	A = I - p*G*D
	#print 'A:\n', A
	
	# use sparse matrix or not:
	if use_sparse:
		raise NotImplementedError("I excluded sparse matrices implementation");
		#AS = sparse.as_sparse_matrix(A)
		#x = cg.conjugate_gradient(AS,e,True)
	else:
		x = np.linalg.solve(A,e)
	return x/np.sum(x)

if __name__ == '__main__':

	# next part is for banchmarking different methods
	file = open('results.txt', 'w')

	# test: generate random matrix:
	for n in np.arange(100, 101, 100):
		print 'n:', n
		file.write(str(n))
		
		#GA = np.where(np.random.randint(0,10, size=(n,n))==0,1,0) # density 10%
		GA = np.where(np.random.randint(0,n, size=(n,n))==0,1,0) # density O(n)
		#GA = np.random.random_integers(0,1, size=(n,n)) # density 50%
		G = np.matrix(GA) # connectivity matrix
		#print G
		density = float(sum(sum(GA)))/n**2
		print 'density:', density

		delta = (1-p)/n
		print 'p:', p, 'delta:', delta
		file.write('\t' + str(density))

		# initial propability distribution:
		R = np.empty(n)
		R.fill(1.0/n)
		R = np.matrix(R.reshape(n,1))
		#print 'R(0):\n', R

		print 'power method:'
		start = time.clock()
		x, e_pm = power_method(R.copy())
		#print x
		print 'iterations:', len(e_pm)
		print 'sum:', np.sum(x)
		total = time.clock() - start
		print 'time:', total
		file.write('\t' + str(total))
		
		print 'iterative:'
		start = time.clock()
		x, e_it = iterative(R.copy())
		#print x
		print 'iterations:', len(e_it)
		print 'sum:', np.sum(x)
		total = time.clock() - start
		print 'time:', total
		file.write('\t' + str(total))
		
		print 'transform to linear equations:'
		print 'using numpy matrices:'
		start = time.clock()
		x = linear_eq(False)
		#print x
		print 'sum:', np.sum(x)
		total = time.clock() - start
		print 'time:', total
		file.write('\t' + str(total) + '\n')
		
		# I excluded this implementation here
		#print 'using sparse matrices:'
		#start = time.clock()
		#x = linear_eq(True)
		#print x
		#print 'sum:', np.sum(x)
		#total = time.clock() - start
		#print 'time:', total
		#file.write('\t' + str(total) + '\n')
		
		print '----------\n'
	
	file.close()
	sys.exit(0)
	
	# write errors to file:
	# only for iterative and power method
	file = open('errors.txt', 'w')
	it = 0
	while True:
		it += 1
		if len(e_it) >= it and len(e_pm) >= it:
			file.write(str(it) + '\t' + str(e_it[it-1]) + '\t' + str(e_pm[it-1]) + '\n')
		elif len(e_it) >= it:
			file.write(str(it) + '\t' + str(e_it[it-1]) + '\t0\n')
		elif len(e_pm) >= it:
			file.write(str(it) + '\t0\t' + str(e_pm[it-1]) + '\n')
		else:
			break
	file.close()
	