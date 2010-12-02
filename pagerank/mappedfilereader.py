#encoding=utf-8
import numpy as np


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
		G = np.matrix(np.zeros(self.n*self.n,dtype=np.uint32).reshape(self.n,self.n)) # adjacency matrix
		for line in f:
			col, row = self._line2el(line)
#			print "col, row",col, row
			G[row,col] = 1; # TODO check indices
		return self.n, G


	# TODO cleanup
	# mapFileName - indexed URLS file
	# mappedFileName - associated indices file
	def __init__(self,mapFileName,mappedFileName, numberOfProcesses):
		f1 = open(mapFileName,'r')
		rows_cols = 0
		for line in f1:
			rows_cols += 1
		f1.close()
#		print "row_cols", rows_cols
#		print "num procs", numberOfProcesses
#		print "row_cols % numProcs", rows_cols % numberOfProcesses

		if (rows_cols % numberOfProcesses != 0): # if number of lines is not divisible by proc. count then make it so
			rows_cols = ( rows_cols/numberOfProcesses + 1) * numberOfProcesses

		self.n = rows_cols
		self.mappedName = mappedFileName
		
		

#------------------------------
"""
mapName = '../data/Map for crawledResults5.txt.txt' 
mappedName = '../data/Mapped version of crawledResults5.txt.txt'

mapName = '../data/Map for crawledResults1.txt.txt' 
mappedName = '../data/Mapped version of crawledResults1.txt.txt'
print "creating reader..."
r = MatReader(mapName, mappedName, 6)
print "reader created"

print "reading..."
n, G = r.read()
print "n=",n
print "reading done"
"""
