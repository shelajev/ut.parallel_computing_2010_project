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

np.random.seed(int(time()))

# Internal utilities
import calculator
import matrixreader as mr
import matrixutils  as mu

def str_td(td):
    return str(td.seconds + td.microseconds/1e6)

class PagerankSolver:
    
    def __init__(self, comm, mapFile, mappedFile):
        self.mapFile = mapFile
        self.mappedFile = mappedFile
        self.checkpoint = 'data/checkpoint'
        
        self.A = None
        self.b = None
        
        self.comm = comm
        self.calculator = calculator.Calculator(self.comm)
        
        self.convergence = 0.00001
        
        self.i = 1
        self.running = False
        self.foundSolution = False
    
    def Setup(self):
        self.calculator.Setup(self.A.shape[0])
    
    def Done(self):
        self.calculator.Done()
    
    def log(self, msg):
        print '       ', msg
    
    def Initialize(self):
        h = self.calculator
        
        self.log('initializing')
        
        # column sums # must be done before tocsr()
        colsum = np.ravel(self.A.sum(axis=0))
        
        self.A = self.A.tocsr()
        self.Setup()
        
        self.b = sparse.csr_matrix(np.ones((self.A.shape[0],1))*1.0)
        
        # distribute data
        self.log('set A')
        h.Set('A', self.A)
        
        self.log('set b')
        h.Set('b', self.b)
        
        self.log('Broadcast colsum')
        h.Broadcast('colsum', colsum)
        
        self.log('prepare PageRank')
        h.PreparePageRank('A', 'A', 'colsum')
        
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
    
    def HasCheckpoint(self):
        return os.path.isfile(self.checkpoint)
    
    def SaveCheckpoint(self):
        self.Save(self.checkpoint)
    
    def LoadCheckpoint(self):
        self.Load(self.checkpoint)
    
    def Load(self, filename):
        # load A, r, rho, w, v, p, x, r_hat, alpha as instance variables
        save = open(filename, "rb")   
        
        # bicgstab scalars
        self.rho = pickle.load(save)
        self.w = pickle.load(save)
        self.alpha = pickle.load(save)
        self.i = pickle.load(save) + 1
        # matrices   
        self.A = pickle.load(save)        
        self.b = pickle.load(save)        
        self.r = pickle.load(save)
        self.v = pickle.load(save)
        self.p = pickle.load(save)
        self.x = pickle.load(save)
        self.r_hat = pickle.load(save)
        
        save.close()

    def Distribute(self):
        self.log('Initializing from saved values')
        self.Setup()
        h = self.calculator
        
        self.A = self.A.tocsr()
        h.Set('A', self.A)
        self.b = self.b.tocsr()
        h.Set('b', self.b)
        
        h.Set('x', self.x.tocsr())
        h.Set('r', self.r.tocsr())
        h.Set('r_hat', self.r_hat.tocsr())
        h.Set('v', self.v.tocsr())
        h.Set('p', self.p.tocsr())
        
        del self.x
        del self.r
        del self.r_hat
        del self.v
        del self.p
            
    def Save(self, filename):
        # collect A, r, rho, w, v, p, x, r_hat, alpha
        # save to file
        h = self.calculator
        save = open(filename, "wb")
        
        pickle.dump(self.rho, save)
        pickle.dump(self.w, save)
        pickle.dump(self.alpha, save)
        pickle.dump(self.i, save)
        
        pickle.dump(h.Collect('A'), save)
        pickle.dump(h.Collect('b'), save)
        pickle.dump(h.Collect('r'), save)
        pickle.dump(h.Collect('v'), save)
        pickle.dump(h.Collect('p'), save)
        pickle.dump(h.Collect('x'), save)
        pickle.dump(h.Collect('r_hat'), save)
        
        save.close()
        
    def bicgstab(self, iterations):
        h = self.calculator
        
        convergence = self.convergence
        alpha = self.alpha
        rho = self.rho
        w = self.w
        
        last_iteration = iterations + self.i
        
        self.running = True
        self.foundSolution = False
        while True:
            # sleep (2)
            self.bicgstabCallback(self.i)
            
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
                self.foundSolution = True
                break

            if (not self.running) or (self.i > last_iteration):
                self.foundSolution = False
                break
            
            self.i += 1
            
        self.rho = rho
        self.alpha = alpha
        self.w = w
    
    def getX(self):
        return self.calculator.Collect('x')
    
    def bicgstabCallback(self, i):
        self.log('iteration %s' % i)
    
    def solve(self):        
        dt1 = datetime.now()
        
        if self.HasCheckpoint():
            self.log('Checkpoint exists, continuing from checkpoint')
            self.LoadCheckpoint()
            dt2 = datetime.now()
            self.Distribute()
        else:
            self.log('Checkpoint does not exist, starting from the beginning...')
            self.A = mr.ReadMatrix(self.mapFile, self.mappedFile)
            self.log(repr(self.A))
            dt2 = datetime.now()
            self.Initialize()
        
        dt3 = datetime.now()
        
        self.bicgstab(10)
        self.SaveCheckpoint()
        
        if self.foundSolution:
            self.log('Found solution.')
        else:
            self.log('Max iterations reached.')
        
        dt4 = datetime.now()
        
        x = self.getX()
        
        x = x / x.sum()
        
        dt5 = datetime.now()
        
        self.Done()
        
        # timings:
        self.log('TIMINGS:')
        self.log('reading input file: ' + str_td(dt2-dt1))
        self.log('distribute values : ' + str_td(dt3-dt2))
        self.log('BiCGStab: ' + str_td(dt4-dt3))
        self.log('collect x: ' + str_td(dt5-dt4))
        self.log('total: ' + str_td(dt5-dt1))
        
        self.log('A.shape: ' + str(self.A.shape))
        self.log('x.shape: ' + str(x.shape))
        self.log('RageRank vector:')
        self.log(x)

def keyboardInputWait(solver, arg2):
    wait = raw_input('Press ENTER to save:\n')
    solver.log('Saving stuff...')
    solver.running = False

def main():
    mapFile = 'data/Map for crawledResults5.txt.txt' 
    mappedFile = 'data/Mapped version of crawledResults5.txt.txt'
    
    comm = MPI.COMM_WORLD
    if comm.rank == 0 :
        s = PagerankSolver(comm, mapFile, mappedFile)
        
        #inputWaiter = Thread(target=keyboardInputWait, args=(s, None))
        #inputWaiter.start()
        
        s.solve()
        s.log('Exiting...')
        
        MPI.Finalize()
        os._exit(0)
    else:
        n = calculator.CalculatorNode(comm)
        n.run()
        MPI.Finalize()
    
if __name__ == "__main__":
    main()
