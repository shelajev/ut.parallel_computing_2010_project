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
from mpi4py import MPI

# Internal utilities
import matrixreader as mr
import matrixutils  as mu

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

#####
# OPERATIONS
# These are the calculation capabilities of the calculator

# distribution
OP_SYNC     = tg()  # this synchronizes all nodes in MPI
OP_SET     = tg()  # distributes a matrix to the nodes
OP_BROADCAST   = tg()  # broadcasts data (not split) to all of the nodes
OP_COLLECT     = tg()  # collects the matrix on the master calculator
OP_COLLECT_SUM = tg()  # collects the total sum of values in the matrix on the master calculator

# creation/manipulation
OP_NEW     = tg()  # creates a new matrix with specified value and size
OP_MOVE    = tg()  # moves a matrix to a different variable
OP_DEL     = tg()  # deletes a matrix from the nodes
OP_OPTIMIZE= tg()  # optimizes specified matrix structure

# computation
OP_ABS     = tg()  # gets the absolute values of a matrix
OP_ADD     = tg()  # adds one matrix to another
OP_SUB     = tg()  # substracts one matrix from another
OP_SCALAR  = tg()  # multiplies a matrix with a scalar
OP_DOT     = tg()  # does a dot product with matrices
OP_MEX     = tg()  # multiplies one matrix with another, both matrices must be the same height

# complex commands
OP_PREPARE_PAGERANK = tg()  # does pagerank preparations

# node internal operations
_OP_CIRCLE = tg()
_OP_CIRCLE_DIR = tg()

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
        at = (mtx.transpose() * a).tocsr()
    return at

#################################
# Calculator Node
#
# does calculations based on the
# commands that Calculator sends it
#################################

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
        self.masks    = {}
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
        """ get my partial matrix (or matrices) """
        a = self.matrixes.get(a)
        b = self.matrixes.get(b)
        c = self.matrixes.get(c)
        
        if b == None:
            return a
        elif c == None:
            return (a,b)
        else:
            return (a,b,c)
    
    def set(self, r, A):
        self.matrixes[r] = A
        if self.masks.has_key(r):
            del self.masks[r]
    
    def getMasked(self, a, mask):
        A = self.get(a)
        return applymask(A, self.rows, mask)
        
    def getMask(self, a):
        """ vector that contains col numbers for each column that contains values and
            0 otherwise"""
        if self.masks.has_key(a):
            z = self.masks[a]
        else:
            A = self.get(a)
            z = mu.ValueColumns(A)
            self.masks[a] = z
        # should do cacheing here
        return z
    
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
        
        R = mu.ListToMatrix(coll)
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
            
        R = mu.ListToMatrix(coll)
        return R
    
    def sync(self):
        self.comm.Barrier()
    
    def send(self, data, dest, tag):
        self.comm.send(data, dest=dest, tag=tag)
        #self.comm.isend(data, dest=dest, tag=tag)

  #################################
  # Matrix Operations
  # 
  # conventions
  #   _opname - unpacks the data and passes it to Opname
  #   Opname  - does the command
  #
  # arguments: 
  #   lower case letters are id-s
  #   capital letters are matrixes
  #
  #   r - is the id of result
  #   a, b - are the argument matrixes
  #################################
  
  #
  # distribution
  #
    
    def Sync(self):
        self.sync()
      
    def Set(self, r, A):
        """ set r value to be A """
        self.set(r, A)
    
    def Broadcast(self, r, A):
        """ sets r to specified data A """
        self.set(r, A)

    def Collect(self, a, target):
        """ collects the matrix a on the master calculator """
        A = self.get(a)
        self.send((self.rows, A), dest = target, tag = OP_COLLECT)
    
    def CollectSum(self, a, target):
        """ collects the total sum of values in the matrix a """
        A = self.get(a)
        self.send(A.sum(), dest = target, tag = OP_COLLECT_SUM)

  #
  # manipulation
  #
        
    def New(self, a, width, value):
        """ create new matrix a with specified width and value """
        A = sparse.csr_matrix(np.ones((self.meight,width))*value)
        self.set(a, A)
        
    def Move(self, r, a):
        """ copies matrix a to r """
        A = self.get(a)
        self.set(r, A.copy())
    
    def Del(self, a):
        """ deletes matrix a from list """
        del self.matrixes[a]
    
    def Optimize(self, a):
        """ optimizes a, if a < 0 optimizes every matrix """
        if a < 0:
            for m in self.matrixes.values():
                if sparse.issparse(m):
                    m.eliminate_zeros()
        else:
            A = self.get(a)
            if sparse.issparse(A):
                A.eliminate_zeros()

  #
  # computation
  #

    def Abs(self, r, a):
        """ set r to absolute value of a"""
        A = self.get(a)
        R = abs(A)
        self.set(r, R)
    
    def Add(self, r, a, b):
        """ adds matrix a and b, puts the result to r """
        A, B = self.get(a,b)
        R = A + B
        self.set(r,R)
        
    def Sub(self, r, a, b):
        """ subtracts matrix b from a, puts the result to r """
        A, B = self.get(a,b)
        R = A - B
        self.set(r,R)
        
    def Scalar(self, r, a, s):
        """ multiplies matrix a with scalar S """
        A = self.get(a)
        R = A * s
        self.set(r,R)
        
    def TensorDot(self, r, a, b):
        """ does a dot product between a and b, puts the result to r """
        A, B = self.get(a,b)
        R = A.multiply(B)
        self.set(r, R)
    
    def Mex(self, r, a, b):
        """ multiplies two matrices a and b, puts the result to r """
        A = self.get(a)
        mask = self.getMask(a)
        B = self.fullMasked(b, mask)
        R = A * B
        self.set(r, R)

  #
  # complex
  #

    def PreparePageRank(self, r, a, c):
        """ inititalizes Pagerank """
        
        A = self.get(a)
        colsum = self.get(c)

        # do A = I - p*A, p = 0.85
        # we want to work with columns
        A = A.tocsc()
        if sparse.isspmatrix_csc(A):
            for j in np.arange(colsum.size):
                if colsum[j] != 0:
                    # divide all elements in that column by colsum[j]:
                    ptr1 = A.indptr[j]
                    ptr2 = A.indptr[j+1]
                    A.data[ptr1:ptr2] /= colsum[j]
        
            p = 0.85
            A = -p*A
        
            # Add 1 to all elements on diaginal:
            A = A.tolil() # because making structural changes to lil_matrix is more efficient
            row = 0
            for col in range(self.rows[0], self.rows[1]):
                A[row, col] += 1.0
                row += 1
        
        self.set(r, A.tocsr())
    
    ##
    # command extractors
    ##
    def _sync(self, data):
        self.sync()
        
    def _set(self, data):
        r, A = data
        self.Set(r, A)
    
    def _broadcast(self, data):
        r, d = data
        self.Broadcast(r, d)
        
    def _collect(self, data):
        a = data
        self.Collect(a, self.master)
        
    def _collect_sum(self, data):
        a = data
        self.CollectSum(a, self.master)

    def _new(self, data):
        a, width, value = data
        self.New(a, width, value)    
    
    def _move(self, data):
        a, b = data
        self.Move(a, b)
    
    def _del(self, data):
        a = data
        self.Del(a)
    
    def _optimize(self, data):
        a = data
        self.Optimize(a)

    def _mex(self, data):
        r, a, v = data
        self.Mex(r,a,v)
    
    def _dot(self, data):
        r, a, b = data
        self.TensorDot(r,a,b)
    
    def _scalar(self, data):
        r,a,s = data
        self.Scalar(r,a,s)
    
    def _add(self, data):
        r,a,b = data
        self.Add(r,a,b)
    
    def _abs(self, data):
        r, a = data
        self.Abs(r,a)        
    
    def _sub(self, data):
        r, a, b = data
        self.Sub(r,a,b)

    def _prepare_pagerank(self, data):
        r, a, c = data
        self.PreparePageRank(r, a, c)


  #################################
  # Main Loops                    #
  #################################
  # First the node will go into Setup mode
  #
  # in setup mode the master should send it
  #   T_HEIGHT - height of data to be stored in total
  #   T_ROWS   - rows that this node should handle
  #   T_SIBLINGS - left and right nodes on a circle
  #   T_DONE     - this completes the setup
  #
  # The node will be in running mode
  # and is able to process commands from master
  #
  
    def run(self):
        """ main loop for running the node """
        self.setup()
        self.loop()
    
    def setup(self):
        """ setup commands """
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
        # map from command tag --> function
        ops = {
            # distribution
            OP_SYNC : self._sync,
            OP_SET  : self._set,
            OP_BROADCAST : self._broadcast,
            OP_COLLECT     : self._collect,
            OP_COLLECT_SUM : self._collect_sum,
            # manipulation
            OP_NEW  : self._new,
            OP_MOVE : self._move,
            OP_DEL  : self._del,
            OP_OPTIMIZE : self._optimize,
            # computation
            OP_ABS  : self._abs,
            OP_ADD  : self._add,
            OP_SUB  : self._sub,
            OP_SCALAR : self._scalar,
            OP_DOT  : self._dot,
            OP_MEX  : self._mex,
            # complex commands
            OP_PREPARE_PAGERANK : self._prepare_pagerank,
        }
        
        comm = self.comm
        s = MPI.Status()
        
        # recv commands until T_DONE
        while True:
            data = comm.recv(None, source=self.master, tag=MPI.ANY_TAG, status=s)
            if s.tag == T_DONE:
                break
            op = ops.get(s.tag)
            if op == None :
                print "Unknown op %s" % s.tag
                os.abort()
            op(data)


#################################
# Calculator
#
# does distributed matrix calculations
#################################

class Calculator:
    """
        This calculator can do matrix calculation on MPI.
        
        This requires at least two nodes to work.
    """
    
    def __init__(self, comm, height = -1):
        """ initializes calculator
            if height is given, it'll also setup CalculatorNodes"""
        self.comm = comm
        # last id
        self.lid = 0        
        # string --> id
        self.matrixes = {}
        # which rows each node holds
        self.rows = {}
        # master rank
        self.master = 0
        # height of the matrices
        self.height = 0
        if height > 0:
            self.Setup(height)
    
    def Setup(self, height):
        """ setup nodes for matrix operations """
        self.height = height
        
        # send height to nodes
        self.Do(self.height, T_HEIGHT)
        
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
        self.Do(0, T_DONE)

    def Done(self):
        """ Kills all the nodes """
        self.Do(0, T_DONE)
        
    def getId(self, name, make_new = True):
        """ get element with name """
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
        
    # network operations
    
    def Sync(self):
        """ Synchronize nodes and master """
        # considering that any new action must be validated from master
        # there should be no syncing error
        self.Do(0, OP_SYNC)
        self.comm.Barrier()
    
    def Do(self, data, tag):
        """ Execute a command on nodes """
        # we can't use bcast
        # data = self.comm.bcast(data, root = self.master, tag = tag)
        for i in range(1, self.comm.size):
            # non blocking send as we are usually just sending
            # commands
            self.comm.isend(data, dest=i, tag=tag)
            #self.Send(data, i, tag)
    
    def Send(self, data, dest, tag):
        """ sends data to dest with tag """
        self.comm.send(data, dest=dest, tag=tag)
        #self.comm.isend(data, dest=dest, tag=tag)

    # distribution
    
    def Set(self, r, A):
        """ distributes matrix A to the nodes with name r """
        a = self.getId(r)
        # loop over
        for i in range(1, self.comm.size):
            rows = self.rows[i]
            s = rows[0]
            e = rows[1]
            self.Send((a, A[s:e,:]), i, OP_SET)
        
    def Collect(self, r):
        """ collects matrix r from the nodes """
        a = self.getId(r, False)
        self.Do(a, OP_COLLECT)
        
        coll = []
        for _ in range(1, self.comm.size):
            tup = self.comm.recv(None, MPI.ANY_SOURCE, OP_COLLECT)
            coll.append(tup)
        
        t = mu.ListToMatrix(coll)        
        return t

    def CollectSum(self, r):
        """ collects the sum of all elements in matrix r """
        a = self.getId(r, False)
        self.Do(a, OP_COLLECT_SUM)
        t = 0.0
        for _ in range(1, self.comm.size):
            t += self.comm.recv(None, MPI.ANY_SOURCE, OP_COLLECT_SUM)
        return t
    
    # matrix operations

    def New(self, name, width, value):
        """ creates a new matrix with 'width' columns and
            all elements set to value """
        a = self.getId(name)
        self.Do((a,width,value), OP_NEW)
    
    def Del(self, name):
        """ deletes matrix """
        if self.matrixes.has_key(name):
            a = self.getId(r, False)
            del self.matrixes[name]
            self.Do(a, OP_DEL)
    
    def Optimize(self, name = False):
        """ optimizes matrix (matrices) structure """
        if not name:
            self.Do(-1, OP_OPTIMIZE)
        else:
            a = self.getId(name, False)
            self.Do(a, OP_OPTIMIZE)
    
    # calculation

    def Scalar(self, r, a, s):
        """ multiplies matrix a with scalar s and stores the result in r """
        self.Do(self.ras(r,a,s), OP_SCALAR)
    
    def Sub(self, r, a, b):
        """ subtracts matrix b from matrix a and stores the result in r """
        self.Do(self.rab(r,a,b), OP_SUB)
        
    def Add(self, r, a, b):
        """ adds matrix b to matrix a and stores the result in r """
        self.Do(self.rab(r,a,b), OP_ADD)
    
    def Move(self, r, a):
        """ copies matrix a to r """
        self.Do(self.ra(r,a), OP_MOVE)
    
    def Dot(self, r, a, b):
        """ calculates dot product of a and b and stores the result in r """
        self.Do(self.rab(r,a,b), OP_DOT)
        return self.CollectSum(r)
    
    def Mex(self, r, a, b):
        """ multiplies matrix a with b and stores the result in r """
        self.Do(self.rab(r,a,b), OP_MEX)
    
    def SumAbs(self, r, a):
        """ calculates the sum of absolute values in a, stores temporary data in r """
        ra = self.ra(r, a)
        self.Do(ra, OP_ABS)
        return self.CollectSum(r)

    def Broadcast(self, r, data):
        """ Broadcasts data to all nodes """
        rid = self.getId(r)
        self.Do((rid, data), OP_BROADCAST)

    def PreparePageRank(self, r, a, c):
        """ Initializes Pagerank, where a is A and c is colsum """
        rab = self.rab(r, a, c)
        self.Do(rab, OP_PREPARE_PAGERANK)
