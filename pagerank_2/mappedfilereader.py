#encoding=utf-8
import numpy as np
from scipy.sparse import * # lil_matrix

class MatReader:
	def _line2el(self,line):
		parts = line.partition("	")
		# [0] and [2] are important, in [2] \n must be stripped
		row = parts[0]
		col = parts[2].strip('\n')
		return int(row), int(col)

	# returns full matrix and the number of lines
	def read(self):

		G = dok_matrix((self.n,self.n))
		f = open(self.mappedName,'r')
		for line in f:
			row, col = self._line2el(line)
			G[row,col]=1
		#return self.n, 	G.tocsc()
		return self.n, 	G.todense()

	"""
	mapFileName - indexed URLS file  ( "i	url\n" )
	mappedFileName - associated indices file ( "i	j\n" )
	"""
	def __init__(self,mapFileName,mappedFileName, numberOfProcesses):
		f1 = open(mapFileName,'r')
		rows_cols = 0
		for line in f1:
			rows_cols += 1
		f1.close()

		if (rows_cols % numberOfProcesses != 0): # if number of lines is not divisible by proc. count then make it so
			rows_cols = ( rows_cols/numberOfProcesses + 1) * numberOfProcesses

		self.n = rows_cols
		self.mappedName = mappedFileName
