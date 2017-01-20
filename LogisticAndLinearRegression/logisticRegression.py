from __future__ import division
import sys
import numpy as np
import random

class LogisticRegression:

	def __init__(self,random_state):
		self.weights=None
		self.intercept=None
		self.random_state=random_state

	def sigmoid(self,z):

		return (1 / (1+np.exp(-z)))

	def fit(self,X,Y):
		row,col=X.shape
		weights=np.array([1.0,1.0,1.0,1.0])
		iterations=0
		error=[]
		
		learning_rate=0.001
		predicted_result=np.zeros((len(X),1))
		bias_val=np.ones((row,1))
		data=np.concatenate((bias_val,X),axis=1)
		np.random.seed(self.random_state)
		while iterations<7000:
			s=np.multiply((np.dot(data,weights)),Y)
			delta_Ein=np.sum((np.multiply(Y.T,data.T) / (1 + np.exp(s)).T).T,axis=0)
			delta_Ein=delta_Ein  / len(data)
			v=(delta_Ein) / np.linalg.norm(delta_Ein)
			weights+=(learning_rate*v)
			iterations=iterations+1
		self.weights=weights[1:]
		self.intercept=weights[0]
		

def main():
	try:
		data=np.loadtxt(sys.argv[1],dtype='float',delimiter=',',usecols=(0,1,2,4))
		X=data[:,0:3]
		Y=data[:,3]
		p=LogisticRegression(random_state=123)
		p.fit(X,Y)
		print "Weights"
		print p.weights
		print "Intercept"
		print p.intercept
	except Exception,e:
		print(str(e))
		print('Syntax:')
		print('\tpython logisticRegression.py <data_filename>')
		sys.exit()
	
	


if __name__ == "__main__":
    main()