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
import matrixreader as mr
import matrixutils as mu

####################
# TAG Enumerator
# calling tg() will give you an unique id
# to be used as an TAG ID
_last_tg = 0xfeed
def tg():
    global _last_tg
    _last_tg += 1
    return _last_tg - 1

#####
# SETUP
# these tag values are used for initial setup
# controller sends those to nodes

T_DONE     = tg()  # Sent when setup or program has completed
T_HEIGHT   = tg()  # this sends the total height of the matrixes stored in matrices
T_ROWS     = tg()  # rows that the node holds
T_SIBLINGS = tg()  # ranks for nodes that are left and right to it
T_MASTER   = tg()  # the master node rank
T_SYNC     = tg()  # this synchronizes all nodes in MPI

#####
# OPERATIONS
# These are the calculation capabilities of the calculator

# manipulation
OP_NEW     = tg()  # creates a new matrix with specified value and size
OP_MOVE    = tg()  # moves a matrix to a different variable
OP_DEL     = tg()  # deletes a matrix from the nodes
OP_OPTIMIZE= tg()  # optimizes specified matrix structure

# computation
OP_ABS     = tg()  # gets the absolute values of a matrix
OP_ADD     = tg()  # adds one matrix to another
OP_SUB     = tg()  # substracts one matrix from another
OP_SCALAR  = tg()  # multiplies a matrix with a scalar
OP_MEX     = tg()  # multiplies one matrix with another, both matrices must be the same height
OP_DOT     = tg()  # does a dot product with matrices

# distribution
OP_SET     = tg()  # distributes a matrix to the nodes
OP_BROADCAST   = tg()  # broadcasts data (not split) to all of the nodes
OP_COLLECT     = tg()  # collects the matrix on the master calculator
OP_COLLECT_SUM = tg()  # collects the total sum of values in the matrix on the master calculator

# complex commands
OP_PREPARE_PAGERANK = tg()  # does pagerank preparations

# node internal operations
_OP_CIRCLE = tg()
_OP_CIRCLE_DIR = tg()

def colmask(a):
    """ vector that contains col numbers for each column that contains values and
        0 otherwise"""
    return mu.ValueColumns(a)

def applymask(a, rows, mask):
    """ applies colmask 'mask' to matrix 'a' limited to 'rows' """
    # this can be probably optimized
    if sparse.isspmatrix_csr(a):
        indptr = np.zeros(a.shape[0] + 1, dtype=int)
        indices = []
        data = []
        
        lastpos=0
        lastt=0
        count = 0
        for i in mask:
            if (i >= rows[0]) and (i < rows[1]):
                t = i - rows[0]
                # set rows between to zero
                indptr[lastt+1:t+1] = count
                # find indices/data range
                indstart, indend = a.indptr[t], a.indptr[t+1]
                # copy col_indices and data
                indices.append(a.indices[indstart:indend])
                data.append(a.data[indstart:indend])
                # set end
                count += indend - indstart
                lastt = t
        indptr[lastt+1:len(indptr)] = count
        data = np.concatenate(data)
        indices = np.concatenate(indices)
        at = sparse.csr_matrix((data,indices,indptr), dtype=a.dtype, shape=a.shape)
    else:
        s = a.shape[0]
        data = np.zeros(s)
        for i in mask:
            if (i >= rows[0]) and (i < rows[1]):
                t = i - rows[0]
                data[t] = 1.0
        mtx = sparse.spdiags(data, [0], s,s)
        at = (mtx * a).tocsr()
    return at

def str_td(td):
    return str(td.seconds + td.microseconds/1e6)

class CalculatorNode:
    """
        Is a node that is able to fulfill matrix operations sent by
        Calculator.
        
        It stores only partial matrixes and those are distributed
        only when neccessary.
    """
    def __init__(self, comm):
        # comm object used for communication
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
        """ collect full matrix with a specified column mask"""
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
        """ set my partial matrix to a value"""
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
        colsum = self.get(c)

        # do A = I - p*A, p = 0.85
        A = A.tocsc() # we want to work with columns
        if sparse.isspmatrix_csc(A):
            for j in np.arange(colsum.size):
                if colsum[j] != 0:
                    # divide all elements in that column by colsum[j]:
                    ptr1 = A.indptr[j]
                    ptr2 = A.indptr[j+1]
                    A.data[ptr1:ptr2] /= colsum[j]
                    pass

            p = 0.85
            A = -p*A

            # Add 1 to all elements on diaginal:
            A = A.tolil() # because making structural changes to lil_matrix is more efficient
            row = 0
            for col in range(self.rows[0], self.rows[1]):
                A[row, col] += 1.0
                row += 1

        self.set(r, A.tocsr())

    def _bcast(self, data):
        r, d = data
        self.Bcast(r, d)

    def Bcast(self, r, vec):
        self.set(r, vec)
    
    def Send(self, data, dest, tag):
        self.comm.send(data, dest=dest, tag=tag)
        #self.comm.isend(data, dest=dest, tag=tag)
    
    def _optimize(self, a):
        for m in self.matrixes.values():
            if sparse.issparse(m):
                m.eliminate_zeros()
    
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
            if s.tag == T_SETUP_DONE:
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
            OP_OPTIMIZE : self._optimize,
            OP_BROADCAST : self._bcast,
            OP_PREPARE_PAGERANK : self._prepare_pagerank,
            T_SYNC : self.Sync
        }
        # start matrix calculation node
        while True:
            data = comm.recv(None, source=self.master, tag=MPI.ANY_TAG, status=s)
            if s.tag == T_DONE:
                MPI.Finalize()
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
    
    def Optimize(self):
        self.Broadcast(0, OP_OPTIMIZE)
    
    def Done(self):
        self.Broadcast(0, T_DONE)
        MPI.Finalize()
    
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
        h.Bcast('colsum', self.colsum)
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
        mapName = '../data/Map for crawledResults1.txt.txt' 
        mappedName = '../data/Mapped version of crawledResults1.txt.txt'
        
        dt1 = datetime.now()
        if os.path.isfile('../data/checkpoint.txt'):
            self.log('Checkpoint file exists, reading...')
            self.Load('../data/checkpoint.txt')
            dt2 = datetime.now()
            self.Distribute()
        else:
            self.log('Checkpoint file does not exist, starting from the beginning...')
            r = mappedfilereader.MatReader(mapName, mappedName)
            s, A = r.read()
            dt2 = datetime.now()
            self.colsum = np.ravel(A.sum(axis=0)) # column sums # must be done before tocsr()
            self.A = A.tocsr()
            self.log(repr(self.A))
            self.b = sparse.csr_matrix(np.ones((s,1))*1.0)
            self.log('s = %d' % s)
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
z