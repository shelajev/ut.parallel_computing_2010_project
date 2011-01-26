#encoding=utf-8
import numpy as np
from scipy.sparse import * # lil_matrix


def ReadFromMappedFile(mappedFilename):
    """ reads a matrix mapped file in format:
            ID_from\tID_to\n
            ...
        and constructs the matrix
    """
    f = open(mappedFileName, 'r') ############### SWITCH BUFFERING ON
    total_rows = 0
    for line in f:
        total_rows += 1
    f.close()
    
    # construct the matrix
    G = dok_matrix((total_rows, total_rows))
    f = open(mappedFilename, "r") ############### SWITCH BUFFERING ON
    for line in f:
        parts = line.split("\t", 1)
        if len(parts) <= 1:
            continue
        row = int(parts[0])
        col = int(parts[1])
        G[row,col] = 1
    
    return G
