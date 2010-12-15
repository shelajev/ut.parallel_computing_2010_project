#encoding=utf-8
import numpy as np
from scipy.sparse import * # lil_matrix

class MatReader:
    def _line2el(self,line):
        parts = line.partition("\t")
        # [0] and [2] are important, in [2] \n must be stripped
        row = parts[0]
        col = parts[2].strip('\n')
        return int(row), int(col)

    def read(self):
        """ returns number of lines and the matrix """
        G = dok_matrix((self.n,self.n))
        f = open(self.mappedName,'r')
        for line in f:
            if line.find('\t') != -1:
                row, col = self._line2el(line)
                G[row,col]=1
        return self.n, G

    def __init__(self,mapFileName, mappedFileName):
        """ initializes for reading
        
        mapFileName - indexed URLS file  ( "i    url\n" )
        mappedFileName - associated indices file ( "i    j\n" )
        """ 
        f1 = open(mapFileName,'r')
        rows_cols = 0
        for line in f1:
            rows_cols += 1
        f1.close()
        
        self.n = rows_cols
        self.mappedName = mappedFileName
