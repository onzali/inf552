from numpy import vectorize
import sys
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt

class LinearRegression:

	def __init__(self):
		self.weights=None
		self.intercept=None

	def fit(self,X,Y): 
		row,col=X.shape
		weights=np.random.rand()
		bias_val=np.ones((row,1))
		X=np.concatenate((bias_val,X),axis=1)
		A=np.linalg.inv(np.dot(X.T,X))
		B=np.dot(X.T,Y)
		weights=np.dot(A,B)
		self.weights=weights[1:]
		self.intercept=weights[0]
		
def main():
	try:
		data=np.loadtxt(sys.argv[1],dtype='float',delimiter=',',usecols=(0,1,2))
		X=data[:,0:2]
		Y=data[:,2]
		p=LinearRegression()
		p.fit(X,Y)
		print "Weights:"
		print p.weights
		print "Intercept:"
		print p.intercept
	except Exception,e:
		print(str(e))
		print('Syntax:')
		print('\tpython linearRegression.py <data_filename>')
		sys.exit()

if __name__ == "__main__":
    main()