import numpy as np
import parallel

TOLERANCE=1E-9
MAX_ITER=1000

def solve(A, b, comm):
    rank = comm.Get_rank()
    master = (rank == 0)
    
    prec = None # For future use :)

    if master:
        # Initial guess:
        x = np.zeros((A.n,1))

        print 'b', b.shape, 'A', (A.m,A.n), 'x', x.shape

        r = b - A*x
    else:
        # Dummy operations are needed as both master and salves have to execute the same methods to use MPI commands
        # However, slaves do not know anything about the real data structures stored on master node.
        A*0 # dummy operation
        r = None

    if master:
        rho_0 = omega_0 = alpha = 1
        p = x.copy() #zero vector
        v = x.copy() #zero vector

    it = 0

    while parallel.error(r, comm) > TOLERANCE and it < MAX_ITER:
        it += 1

        print 'Rank: %d, iteration: %d' % (rank,it)

        if master:
            #print 'Iteration:', it

            if prec:
                z = prec(r)
            else:
                z = r.copy()

            print 'Step 1'
            rho_1 = parallel.dot(z,r,comm) #1
        else:
            parallel.dot(0,0,comm) # dummy

        if master:
            print 'Step 2'
            beta = (rho_1/rho_0)*(alpha/omega_0) #2
            print 'Step 3'
            p = r + beta*(p - omega_0*v) #3
            print 'Step 4'
            v = A*p #4
        else:
            A*0 # dummy
            

        if master:
            print 'Step 5'
            alha = rho_1/parallel.dot(z,v,comm) #5
        else:
            parallel.dot(0,0,comm) # dummy

        if master:
            print 'Step 6'
            s = r - alpha*v #6
            print 'Step 7'
            t = A*s #7
        else:
            A*0 # dummy

        if master:
            print 'Step 8'
            omega_1 = parallel.dot(t,s,comm)/parallel.dot(t,t,comm) #8
        else:
            parallel.dot(0,0,comm) # dummy
            parallel.dot(0,0,comm) # dummy

        if master:
            print 'Step 9'
            x = x + alpha*p + omega_1*s #9

            print 'Step 11'
            r = s - omega_1*t # 11

            omega_0 = omega_1
            rho_0 = rho_1

    if master:
        return x
    return


def error(r):
    return np.sqrt(np.sum(np.power(r,2)))

