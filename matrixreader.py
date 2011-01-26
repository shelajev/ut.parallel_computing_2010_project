#encoding=utf-8
import numpy as np
from scipy.sparse import * # lil_matrix

import mmap

def ReadMatrix(mappedFilename):
    """ reads a matrix mapped file in format:
            ID_from\tID_to\n
            ...
        and constructs the matrix
    """
    f = open(mappedFilename, 'r')
    buffer = mmap.mmap(f.fileno(), 0, prot = mmap.PROT_READ)
    
    readline = buffer.readline
    
    # count the number of rows
    total_rows = 0
    while readline():
        total_rows += 1
    
    buffer.seek(0)
    
    # construct the matrix
    G = dok_matrix((total_rows, total_rows))
    line = readline()
    while line:
        parts = line.split("\t", 1)
        line = readline()
        if len(parts) <= 1:
            continue
        row = int(parts[0])
        col = int(parts[1])
        G[row,col] = 1
    
    buffer.close()
    f.close()
    
    return G

def ReadLinks(mapFilename):
    """ reads in a id with link
            ID\tlink\n
            ...
        returns a dictionary with ID as key and value as link
    """
    
    f = open(mapFilename, 'r', buffering=262144)
    items = {}
    for line in f:
        parts = line.split("\t", 1)
        if len(parts) <= 1:
            continue
        id = int(parts[0])
        link = parts[1].strip()
        items[id] = link
    f.close()
    
    return items