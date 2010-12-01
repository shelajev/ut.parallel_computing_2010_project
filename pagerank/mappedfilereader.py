#encoding=utf-8
import numpy as np

# TODO 1 test this class read() locally
# TODO 2 test this class read() globally
""" ../data/
crawledResults1.txt 
crawledResults5.txt 
Map for crawledResults1.txt.txt 
Map for crawledResults5.txt.txt 
Mapped version of crawledResults1.txt.txt 
Mapped version of crawledResults5.txt.txt 
"""


class MatReader:
	def _line2el(self,line):
		parts = line.partition("	")
		# [0] and [2] are important, in [2] \n must be stripped
		row = parts[0]
		col = parts[2].strip('\n')
		val = 1
		return col, row

		# return full matrix
	def read(self):
		f = open(self.mappedName,'r')
		G = np.matrix(np.zeros(self.n*self.n,dtype=np.uint32).reshape(n,n)) # adjacency matrix
		for line in f:
			col, row = self._line2el(line)
			print "col, row",col, row
			G[int(c)-1,int(id)-1] = 1; # TODO check indices
		return self.n, G


	# TODO cleanup
	# mapFileName - indexed URLS file
	# mappedFileName - associated indices file
	def __init__(self,mapFileName,mappedFileName):
		f1 = open(mapFileName,'r')
		rows_cols = 0
		for line in f1:
			rows_cols += 1
		f1.close()
		f2 = open(mappedFileName, 'r')
		
		vals = 0
		for line in f2:
			vals += 1
		f2.close()

		rows = rows_cols
		cols = rows_cols
		
		self.n = rows_cols
		print "cols rows vals",cols,rows,vals
		self.mappedName = mappedFileName
		
		

#------------------------------
"""
mapName = 'data/Map for crawledResults5.txt.txt' 
mappedName = 'data/Mapped version of crawledResults5.txt.txt'

r = MatReader(mapName, mappedName)
A = r.read()
"""
