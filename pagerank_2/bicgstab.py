#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sparse
from numpy.random import randn, rand
import random
from mpi4py import MPI
from time import time, sleep
import os
import mappedfilereader
import pickle
from threading import Thread

# todo
    # bicgstab has quite large error, can this be improved?

# tag enumeration
_last_tg = 0xfeed
def tg():
    global _last_tg
    _last_tg += 1
    return _last_tg - 1

# setup
T_DONE = tg()
T_HEIGHT = tg()
T_ROWS = tg()
T_SIBLINGS = tg()
T_MASTER = tg()
T_SYNC = tg()

# operations
OP_NEW    = tg()
OP_SCALAR = tg()
OP_SUB    = tg()
OP_ADD    = tg()
OP_MOVE   = tg()
OP_MEX    = tg()
OP_ABS    = tg()
OP_SET    = tg()
OP_DOT    = tg()
OP_COLLECT_SUM = tg()
OP_COLLECT = tg()
OP_BROADCAST  = tg()
OP_PREPARE_PAGERANK = tg()

# internal operations
_OP_CIRCLE = tg()
_OP_CIRCLE_DIR = tg()

def listToMatrix(mtxs):
    """ converts list of [ (rows, matrix) ], to a matrix  """
    mtxs.sort(key = lambda x : x[0])
    mtxs = map(lambda x : x[1], mtxs)
    R = sparse.vstack(mtxs)
    return R

def uniqlist(a):
    """ returns unique elements from list """
    return {}.fromkeys(a).keys()

def colmask(a):
    """ vector that contains col numbers for each column that contains values and
        0 otherwise"""
    # this can be probably optimized
    # returns nonzero indices of columns
    M = a.nonzero()[1]
    # remove duplicates and sort
    M = sorted(uniqlist(M))    
    return M

def applymask(a, rows, mask):
    """ applies colmask 'mask' to matrix 'a' limited to 'rows' """
    # this can be probably optimized
    s = a.shape[0]
    data = np.zeros(s)
    for i in mask:
        if (i >= rows[0]) and (i < rows[1]):
            t = i - rows[0]
            data[t] = 1.0
    mtx = sparse.spdiags(data, [0], s,s)
    at = (a.transpose() * mtx).transpose().tocsr()
    return at

class CalculatorNode:
    """
        Is a node that is able to fulfill matrix operations sent by
        RemoteMatrixHandler.
        
        It stores only partial matrixes and those are distributed
        only when neccessary.
    """
    def __init__(self, comm):
        self.comm = comm
        # matrix columns
        # id --> partial_matrix
        self.matrixes = {}
        # which rows am I holding 
        self.rows = (0,0)
        # my partial matrix height
        self.meight = 0
        # the height of the full matrix
        self.height = 0
        # assume
        self.master = 0
        # siblings = left, right
        self.left = 0
        self.right = 0
        self.id = 0
        
    def get(self, a, b = -1, c = -1):
        """ get my partial matrix """
        a = self.matrixes.get(a)
        b = self.matrixes.get(b)
        c = self.matrixes.get(c)
        
        if b == None:
            return a
        elif c == None:
            return (a,b)
        else:
            return (a,b,c)
    
    def full(self, a):
        """ collect full matrix """
        mine = self.matrixes[a]
        rows = self.rows
        height = self.height
        
        send = (rows,mine)
        coll = [send]
        # use circular distribution
        for _ in range(1, self.comm.size - 1):
            data = self.comm.sendrecv(send, self.left,  _OP_CIRCLE,
                                      None, self.right, _OP_CIRCLE)
            coll.append(data)
            send = data
        
        R = listToMatrix(coll)
        return R

    def fullMasked(self, a, mask):
        """ collect full matrix """
        mine = self.matrixes[a]
        rows = self.rows
        height = self.height
        
        send = (self.id, mask)
        coll = [(rows,mine)]
        # use circular distribution
        for _ in range(1, self.comm.size - 1):
            # pass (id, mask) around
            data = self.comm.sendrecv(send, self.left,  _OP_CIRCLE,
                                      None, self.right, _OP_CIRCLE)
            send = data
            # apply 'mask' to this nodes data and send it to 'id'
            masked  = applymask(mine, rows, data[1])
            mtx = self.comm.sendrecv((rows, masked), data[0], _OP_CIRCLE_DIR,
                                      None, MPI.ANY_SOURCE , _OP_CIRCLE_DIR)
            coll.append(mtx)
            
        R = listToMatrix(coll)
        return R

        
    def set(self, r, R):
        """ set my partial matrix """
        self.matrixes[r] = R
    
    def _new(self, data):
        a, width, value = data
        self.New(a, width, value)
    
    def New(self, a, width, value):
        A = sparse.csr_matrix(np.ones((self.meight,width))*value)
        self.set(a, A)
    
    def _collect(self, data):
        a = data
        self.Collect(a, self.master)
    
    def Collect(self, a, target):
        A = self.get(a)
        self.Send((self.rows, A), dest = target, tag = OP_COLLECT)
    
    def _collect_sum(self, data):
        a = data
        self.CollectSum(a, self.master)
    
    def CollectSum(self, a, target):
        A = self.get(a)
        self.Send(A.sum(), dest = target, tag = OP_COLLECT_SUM)
    
    def _mex(self, data):
        r, a, v = data
        self.Mex(r,a,v)
    
    def Mex(self, r, a, v):
        A = self.get(a)
        mask = colmask(A)
        V = self.fullMasked(v, mask)
        R = A * V
        self.set(r, R)
    
    def _dot(self, data):
        r, a, b = data
        self.TensorDot(r,a,b)
        
    def TensorDot(self, r, a, b):
        A, B = self.get(a,b)
        R = A.multiply(B)
        self.set(r, R)
    
    def _scalar(self, data):
        r,a,s = data
        self.Scalar(r,a,s)
        
    def Scalar(self, r, a, s):
        A = self.get(a)
        R = A * s
        self.set(r,R)
    
    def _add(self, data):
        r,a,b = data
        self.Add(r,a,b)
    
    def Add(self, r, a, b):
        A, B = self.get(a,b)
        R = A + B
        self.set(r,R)
    
    def _sub(self, data):
        r, a, b = data
        self.Sub(r,a,b)
        
    def Sub(self, r, a, b):
        A, B = self.get(a,b)
        R = A - B
        self.set(r,R)
    
    def _abs(self, data):
        r, a = data
        self.Abs(r,a)
    
    def Abs(self, r, a):
        A = self.get(a)
        R = abs(A)
        self.set(r, R)
    
    def _move(self, data):
        a, b = data
        self.Move(a, b)
    
    def _set(self, data):
        r, A = data
        self.Set(r, A)
        
    def Set(self, r, A):
        self.set(r, A)
    
    def Sync(self, data):
        self.comm.Barrier()
    
    def Move(self, a, b):
        B = self.get(b)
        self.set(a, B.copy())

    def _prepare_pagerank(self, data):
        r, a, c = data
        self.PreparePageRank(r, a, c)

    def PreparePageRank(self, r, a, c):
        A = self.get(a)
        #A = A.tolil()
        colsum = self.get(c)

        for j in np.arange(colsum.size):
            if colsum[j] != 0:
                # divide all elements in that column by colsum[j]:
                #A[:,j] /= colsum[j]
                pass

        # do A = I - p*A, p = 0.85

    def _bcast(self, data):
        r, d = data
        self.Bcast(r, d)

    def Bcast(self, r, vec):
        self.set(r, vec)
    
    def Send(self, data, dest, tag):
        self.comm.send(data, dest=dest, tag=tag)
        #self.comm.isend(data, dest=dest, tag=tag)
    
    # mainloop
    
    def run(self):
        self.setup()
        self.loop()
    
    def setup(self):
        comm = self.comm
        s = MPI.Status()
        # start matrix calculation node
        self.id = self.comm.rank        
        while True:
            data = comm.recv(None, self.master, MPI.ANY_TAG, s)
            if s.tag == T_DONE:
                break
            if s.tag == T_HEIGHT:
                self.height = data
            elif s.tag == T_ROWS:
                self.rows = data
                self.meight = self.rows[1] - self.rows[0]
            elif s.tag == T_SIBLINGS:
                self.left, self.right = data
            else:
                print "Unknown tag %s" % s.tag
                os.abort()
    
    def loop(self):
        comm = self.comm
        s = MPI.Status()
        ops = {
            OP_NEW  : self._new,    
            OP_SCALAR : self._scalar,
            OP_SUB  : self._sub,
            OP_ADD  : self._add,
            OP_MOVE : self._move,
            OP_MEX  : self._mex,
            OP_ABS  : self._abs,
            OP_SET  : self._set,
            OP_DOT  : self._dot,
            OP_COLLECT_SUM : self._collect_sum,
            OP_COLLECT     : self._collect,
            OP_BROADCAST : self._bcast,
            OP_PREPARE_PAGERANK : self._prepare_pagerank,
            T_SYNC : self.Sync
        }
        # start matrix calculation node
        while True:
            data = comm.recv(None, source=self.master, tag=MPI.ANY_TAG, status=s)
            if s.tag == T_DONE:
                break
            op = ops.get(s.tag)
            if op == None :
                print "Unknown op %s" % s.tag
                os.abort()
            op(data)

class Calculator:
    def __init__(self, comm, height):
        self.comm = comm
        # last id
        self.lid = 0        
        # string --> id
        self.matrixes = {}
        self.rows = {}
        self.master = 0
        self.height = height
        self.Setup()
    
    def Setup(self):
        """ setup nodes for matrix operations """
        # send height to nodes
        self.Broadcast(self.height, T_HEIGHT)
        
        # send rows to nodes
        b = 0
        left = self.height
        for i in range(1, self.comm.size):
            count = left / (self.comm.size - i)
            rows = (b, b + count)
            self.rows[i] = rows
            b = rows[1]
            left -= count
            self.Send(rows, dest=i, tag=T_ROWS)
        
        # send (left, right) to nodes
        for i in range(1, self.comm.size):
            left = (i - 1 + self.comm.size) % self.comm.size
            if left == 0: left = self.comm.size - 1
            right = (i + 1) % self.comm.size
            if right == 0: right = 1
            self.Send((left,right), dest=i, tag=T_SIBLINGS)
        
        # send setup done
        self.Broadcast(0, T_DONE)
    
    def getId(self, name, make_new = True):
        a = self.matrixes.get(name)
        if a == None:
            if not make_new:
                raise Exception("Value '%s' not in nodes." % name)
            a = self.lid
            self.lid += 1
        self.matrixes[name] = a
        return a
    
    # helper functions that
    # generate id, if neccesary and return tuple for sending over
    def ras(self, r, a, s):
        """ r - result, a - matrix, s - scalar """
        rid = self.getId(r)
        aid = self.getId(a, False)
        return (rid, aid, s)
    
    def rab(self, r, a, b):
        """ r - result, a - matrix, b - matrix """
        rid = self.getId(r)
        aid = self.getId(a, False)
        bid = self.getId(b, False)
        return (rid, aid, bid)
    
    def ra(self, r, a):
        """ r - result, a - matrix, b - matrix """
        rid = self.getId(r)
        aid = self.getId(a, False)
        return (rid,aid)
    
    # matrix operations

    def New(self, name, width, value):
        a = self.getId(name)
        self.Broadcast((a,width,value), OP_NEW)
        
    def Scalar(self, r, a, s):
        self.Broadcast(self.ras(r,a,s), OP_SCALAR)
    
    def Sub(self, r, a, b):
        self.Broadcast(self.rab(r,a,b), OP_SUB)
        
    def Add(self, r, a, b):
        self.Broadcast(self.rab(r,a,b), OP_ADD)
    
    def Move(self, r, a):
        self.Broadcast(self.ra(r,a), OP_MOVE)
    
    def Mex(self, r, a, b):
        self.Broadcast(self.rab(r,a,b), OP_MEX)
    
    def SumAbs(self, r, a):
        ra = self.ra(r, a)
        self.Broadcast(ra, OP_ABS)
        return self.CollectSum(r)
    
    def Set(self, r, A):
        a = self.getId(r)
        # loop over
        for i in range(1, self.comm.size):
            rows = self.rows[i]
            s = rows[0]
            e = rows[1]
            self.Send((a, A[s:e,:]), i, OP_SET)

    def Bcast(self, r, data):
        a = self.getId(r)
        for i in range(1, self.comm.size):
            self.Send((a, data), i, OP_BROADCAST)

    def PreparePageRank(self, r, a, c):
        rab = self.rab(r, a, c)
        self.Broadcast(rab, OP_PREPARE_PAGERANK)
    
    def Dot(self, r, a, b):
        self.Broadcast(self.rab(r,a,b), OP_DOT)
        return self.CollectSum(r)
    
    def CollectSum(self, r):
        a = self.getId(r, False)
        self.Broadcast(a, OP_COLLECT_SUM)
        t = 0.0
        for _ in range(1, self.comm.size):
            t += self.comm.recv(None, MPI.ANY_SOURCE, OP_COLLECT_SUM)
        self.Sync()
        return t
    
    def Collect(self, r):
        a = self.getId(r, False)
        self.Broadcast(a, OP_COLLECT)
        
        coll = []
        for _ in range(1, self.comm.size):
            tup = self.comm.recv(None, MPI.ANY_SOURCE, OP_COLLECT)
            coll.append(tup)
        
        t = listToMatrix(coll)
        self.Sync()
        
        return t
    
    def Done(self):
        self.Broadcast(0, T_DONE)
    
    # network operations
    
    def Sync(self):
        # considering that any new action must be validated from master
        # there should be no syncing error
        # self.Broadcast(0, T_SYNC)
        # self.comm.Barrier()
        pass
    
    def Broadcast(self, data, tag):
        # we can't use bcast
        # data = self.comm.bcast(data, root = self.master, tag = tag)
        for i in range(1, self.comm.size):
            # non blocking send as we are usually just sending
            # commands
            self.comm.isend(data, dest=i, tag=tag)
            #self.Send(data, i, tag)
    
    def Send(self, data, dest, tag):
        self.comm.send(data, dest=dest, tag=tag)
        #self.comm.isend(data, dest=dest, tag=tag)

class SolverDistributed:
    def __init__(self, comm):
        self.A = None
        self.b = None
        self.comm = comm
        self.convergence = 0.00001
        self.callback = None
        self.running = False
        self.i = 1
    
    def Setup(self):
        self.calculator = Calculator(self.comm, self.A.shape[0])
        
    def log(self, msg):
        print '       ', msg
    
    def Initialize(self):
        self.log('initializing')
        h = self.calculator
        
        # distribute data
        self.log('set A')
        h.Set('A', self.A)

        self.log('bcast colsum')
        h.Bcast('colsum', self.colsum)

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
        # distributing is done using Initialize ?
        
        save = open(filename, "rb")   
        self.log('Loading A')
        A = pickle.load(save)  
        self.A = A.tocsr()

        self.b = sparse.csr_matrix(np.ones((A.shape[0],1))*1.0)

        self.log('Loading r')
        r = pickle.load(save)
        self.r = r
        self.log('Loading rho')
        rho = pickle.load(save)
        self.rho = rho
        self.log('Loading w')
        w = pickle.load(save)
        self.w = w
        self.log('Loading v')
        v = pickle.load(save)
        self.v = v
        self.log('Loading p')
        p = pickle.load(save)
        self.p = p
        self.log('Loading x')
        x = pickle.load(save)
        self.x = x
        self.log('Loading r_hat')
        r_hat = pickle.load(save)
        self.r_hat = r_hat
        self.log('Loading alpha')
        alpha = pickle.load(save)
        self.alpha = alpha
        self.log('Loading i')
        self.i = pickle.load(save) + 1
        save.close()
    
    def Save(self, filename):
        # collect A, r, rho, w, v, p, x, r_hat, alpha
        # save to file
        h = self.calculator
        save = open(filename, "wb")   
        self.log('Saving A')
        pickle.dump(h.Collect('A'), save)
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
        mapName = '../data/Map for crawledResults1.txt.txt' 
        mappedName = '../data/Mapped version of crawledResults1.txt.txt'
        
        if os.path.isfile('../data/checkpoint.txt'):
            self.log('Checkpoint file exists, reading...')
            self.Load('../data/checkpoint.txt')
            self.Setup() 
            self.Initialize()
        else:
            self.log('Checkpoint file does not exist, starting from the beginning...')
            r = mappedfilereader.MatReader(mapName, mappedName)
            s, A = r.read()
            self.colsum = np.ravel(A.sum(axis=0)) # column sums # must be done before tocsr()
            self.A = A.tocsr()
            self.log(repr(self.A))
            self.b = sparse.csr_matrix(np.ones((s,1))*1.0)
            self.log('s = %d' % s)
            self.Setup()
            self.Initialize()
        
        self.bicgstab(10)
        x = self.getX()
        x_i = self.calculator.Collect('x')
        self.log(self.A.shape)
        self.log(x.shape)
        self.log(x.todense()[3:10,:])
        z = self.A*x
        self.log(sum(abs(z.todense() - self.b.todense())))
        self.Done()

def saveCall(solver, arg2):
    wait = raw_input('Press ENTER to save:\n')
    solver.log('Saving stuff...')
    solver.running = False

def main():
    comm = MPI.COMM_WORLD
    if comm.rank == 0 :
        s = SolverDistributed(comm)
        save = Thread(target=saveCall, args=(s, None))
        save.start()
        s.testSolver2()
        s.log('Exiting...')
        os._exit(0)
    else:
        n = CalculatorNode(comm)
        n.run()

if __name__ == "__main__":
    main()
