#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Utilities
import os
from threading import Thread
from time import time, sleep
from datetime import datetime
import pickle

# Libraries
import numpy as np
import scipy.sparse as sparse
from numpy.random import randn, rand
from mpi4py import MPI

# Internal utilities
import calculator
import matrixreader as mr
import matrixutils  as mu

class BicgstabSolver:
    
    def __init__(self, comm):
        self.A = None
        self.b = None
        self.comm = comm
        self.convergence = 0.00001
        self.callback = None
        self.running = False
        self.i = 1
    
    def Setup(self):
        self.calculator = calculator.Calculator(self.comm, self.A.shape[0])
        
    def log(self, msg):
        print '       ', msg
    
    def Initialize(self):
        self.log('initializing')
        h = self.calculator
        
        # distribute data
        self.log('set A')
        h.Set('A', self.A)
        del self.A

        self.log('bcast colsum')
        h.Broadcast('colsum', self.colsum)

        self.log('prepare PageRank')
        h.PreparePageRank('A', 'A', 'colsum')

        self.log('set b')
        h.Set('b', self.b)
        self.log('create x')
        h.New('x', 1, 1.0)
        
        self.log('calculate : r = b - mex(A,x)')
        # r = b - mex(A, x)
        h.Mex('Ax', 'A', 'x')
        h.Sub('r', 'b', 'Ax')
        
        self.log('calculate : r_hat = r.copy()')
        # r_hat = r.copy()
        h.Move('r_hat', 'r')
        
        self.rho = self.alpha = self.w = 1.0
        self.alpha = 1.0
        
        self.log('calculate : v = p = ->1')
        h.New('v', 1, 1.0)
        h.Move('p', 'v')
    
    def Load(self, filename):
        # load A, r, rho, w, v, p, x, r_hat, alpha as instance variables
        save = open(filename, "rb")   
        self.log('Loading A')
        A = pickle.load(save)  
        self.A = A.tocsr()
        self.b = sparse.csr_matrix(np.ones((A.shape[0],1))*1.0)
        self.log('Loading colsum')
        self.colsum = pickle.load(save)
        self.log('Loading r')
        self.r = pickle.load(save)
        self.log('Loading rho')
        self.rho = pickle.load(save)
        self.log('Loading w')
        self.w = pickle.load(save)
        self.log('Loading v')
        self.v = pickle.load(save)
        self.log('Loading p')
        self.p = pickle.load(save)
        self.log('Loading x')
        self.x = pickle.load(save)
        self.log('Loading r_hat')
        self.r_hat = pickle.load(save)
        self.log('Loading alpha')
        self.alpha = pickle.load(save)
        self.log('Loading i')
        self.i = pickle.load(save) + 1
        save.close()

    def Distribute(self):
        self.log('Initializing from saved values')
        self.calculator = Calculator(self.comm, self.A.shape[0])
        h = self.calculator

        h.Set('A', self.A)
        h.Broadcast('colsum', self.colsum)
        #h.PreparePageRank('A', 'A', 'colsum')
        h.Set('b', self.b)

        h.Set('x', self.x.tocsr())        
        h.Set('r', self.r.tocsr())
        h.Set('r_hat', self.r_hat.tocsr())
        h.Set('v', self.v.tocsr())
        h.Set('p', self.p.tocsr())
            
    def Save(self, filename):
        # collect A, r, rho, w, v, p, x, r_hat, alpha
        # save to file
        h = self.calculator
        save = open(filename, "wb")   
        self.log('Saving A')
        pickle.dump(h.Collect('A'), save)
        self.log('Saving colsum')
        pickle.dump(self.colsum, save)
        self.log('Saving r')
        pickle.dump(h.Collect('r'), save)
        self.log('Saving rho')
        pickle.dump(self.rho, save)
        self.log('Saving w')
        pickle.dump(self.w, save)
        self.log('Saving v')
        pickle.dump(h.Collect('v'), save)
        self.log('Saving p')
        pickle.dump(h.Collect('p'), save)
        self.log('Saving x')
        pickle.dump(h.Collect('x'), save)
        self.log('Saving r_hat')
        pickle.dump(h.Collect('r_hat'), save)
        self.log('Saving alpha')
        pickle.dump(self.alpha, save)
        self.log('Saving i')
        pickle.dump(self.i, save)
        save.close()
        
    def bicgstab(self, iterations):
        h = self.calculator
        
        convergence = self.convergence
        alpha = self.alpha
        rho = self.rho
        w = self.w
        
        self.running = True
        while True:
            # sleep (2)
            self.log('iteration %s' % self.i)
            
            rho_i = h.Dot('rho_i', 'r_hat', 'r')
            beta = (rho_i / rho) * (alpha / w)
            
            # p_i = r + beta * (p - w * v)
            h.Scalar('w*v', 'v', w)
            h.Sub('p - w*v', 'p', 'w*v')
            h.Scalar('beta * (p - w*v)', 'p - w*v', beta)
            h.Sub('p_i', 'r', 'beta * (p - w*v)')
            
            ##
            h.Mex('v_i', 'A', 'p_i')
            alpha = rho_i / h.Dot('_0', 'r_hat', 'v_i')
            
            # s = r - alpha * v_i
            h.Scalar('alpha * v_i', 'v_i', alpha)
            h.Sub('s', 'r', 'alpha * v_i')
            
            ##
            h.Mex('t', 'A', 's')
            w_i = h.Dot('_0', 't', 's') / h.Dot('_0', 't', 't')
            
            # x_i = x + alpha * p_i + w_i * s
            h.Scalar('w_i * s', 's', w_i)
            h.Scalar('alpha * p_i', 'p_i', alpha)
            h.Add('x_i', 'x', 'alpha * p_i')
            h.Add('x_i', 'x_i', 'w_i * s')
            
            # if (abs(x_i - x)).sum() < convergence
            h.Sub('diff', 'x_i', 'x')
            s = h.SumAbs('_0', 'diff')
            
            # r_i = s - w_i * t
            h.Scalar('w_i * t', 't', w_i)
            h.Sub('r_i', 's', 'w_i * t')
            
            # shift for next iteration
            rho = rho_i
            w = w_i
            
            h.Move('r', 'r_i')
            h.Move('v', 'v_i')
            h.Move('p', 'p_i')
            h.Move('x', 'x_i')
            
            if s < convergence:
                self.log('The right solution found!')
                break
            if self.i >= iterations:
                self.log('Maximum number of iterations reached, convergence not reached. Saving...')
                self.Save('../data/checkpoint.txt')
                break
            if self.callback != None:
                self.callback(i)
            if not self.running:
                self.Save('../data/checkpoint.txt')
                break
            self.i += 1
            
        self.rho = rho
        self.alpha = alpha
    
    def getX(self):
        return self.calculator.Collect('x')
    
    def Done(self):
        self.calculator.Done()

    def testSolver(self):
        np.random.seed(int(time()))
        s = self.comm.size * 10 + 3
        self.A = np.asmatrix(rand(s,s))*20
        self.b = np.asmatrix(rand(s,1))*20
        self.Setup()
        self.Initialize()
        self.bicgstab(1000)
        x = self.getX()
        x_i = self.calculator.Collect('x')
        z = self.A*x
        self.log(sum(abs(z - self.b)))
        self.Done()

    def testSolver2(self):
        np.random.seed(int(time()))
        # set input files
        mapName = 'data/Map for crawledResults1.txt.txt' 
        mappedName = 'data/Mapped version of crawledResults1.txt.txt'
        
        dt1 = datetime.now()
        if os.path.isfile('checkpoint/checkpoint.txt'):
            self.log('Checkpoint file exists, reading...')
            self.Load('checkpoint/checkpoint.txt')
            dt2 = datetime.now()
            self.Distribute()
        else:
            self.log('Checkpoint file does not exist, starting from the beginning...')
            A = mr.ReadMatrix(mapName, mappedName)
            dt2 = datetime.now()
            self.colsum = np.ravel(A.sum(axis=0)) # column sums # must be done before tocsr()
            self.A = A.tocsr()
            self.log(repr(self.A))
            self.b = sparse.csr_matrix(np.ones((A.shape[0],1))*1.0)
            self.log('s = %d' % A.shape[0])
            self.Setup()
            self.Initialize()
        
        dt3 = datetime.now()
        self.bicgstab(10)
        dt4 = datetime.now()
        x = self.getX()
        x =x/x.sum()
        dt5 = datetime.now()
        #self.log(x.todense()[3:10,:])
        #z = self.A*x
        #self.log(sum(abs(z.todense() - self.b.todense())))
        self.Done()

        # timings:
        self.log('TIMINGS:')
        self.log('reading input file: ' + str_td(dt2-dt1))
        self.log('distribute values: ' + str_td(dt3-dt2))
        self.log('BiCGStab: ' + str_td(dt4-dt3))
        self.log('collect x: ' + str_td(dt5-dt4))
        self.log('total: ' + str_td(dt5-dt1))

        self.log('A.shape: ' + str(self.A.shape))
        self.log('x.shape: ' + str(x.shape))
        self.log('RageRank vector:')
        self.log(x)

def saveCall(solver, arg2):
    wait = raw_input('Press ENTER to save:\n')
    solver.log('Saving stuff...')
    solver.running = False

def main():
    comm = MPI.COMM_WORLD
    if comm.rank == 0 :
        s = BicgstabSolver(comm)
        save = Thread(target=saveCall, args=(s, None))
        save.start()
        s.testSolver2()
        s.log('Exiting...')
        os._exit(0)
    else:
        n = calculator.CalculatorNode(comm)
        n.run()
    
    MPI.Finalize()
    
if __name__ == "__main__":
    main()