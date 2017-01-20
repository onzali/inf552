from numpy import vectorize
import sys
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt

class Pocket:

	def __init__(self,random_state=None):
		self.numberOfIter=7000
		self.minWeights=None
		self.intercept=None
		self.errorCountArr=np.zeros(7000)
		self.minErrors=7000
		self.random_state=random_state

	def predict(self,z):
	    if z<0:
	        return -1
	    else:
	        return 1

	def checkPredictedValue(self,z,actualZ):
	    if(z==actualZ):
	        return True
	    else:
	        return False

	def fit(self,X,Y):
	    
	    row,col=X.shape
	    weights=np.array([1.0,1.0,1.0,1.0]) 
	    vpredict = vectorize(self.predict)
	    vcheckPredictedValue=vectorize(self.checkPredictedValue)
	    learning_rate=1.0
	    predicted_result=np.zeros((len(X),1))
	    bias_val=np.ones((row,1))
	    data=np.concatenate((bias_val,X),axis=1)
	    np.random.seed(self.random_state)
	    iter=0
	    while self.numberOfIter>0:    
	        weightedSum=np.dot(data,weights)
	        predictedValues=vpredict(weightedSum)
	        predictions=vcheckPredictedValue(predictedValues,Y)
	        misclassifiedPoints=np.where(predictions==False)
	        misclassifiedPoints=misclassifiedPoints[0]
	        numOfErrors=len(misclassifiedPoints)
	        self.errorCountArr[iter]=numOfErrors
	        if numOfErrors<self.minErrors:
	        	self.minErrors=numOfErrors
	        iter+=1
	        misclassifiedIndex=np.random.choice(misclassifiedPoints)
	        weights+=(Y[misclassifiedIndex]*learning_rate*data[misclassifiedIndex])
	        self.numberOfIter-=1
	    self.weights=weights[1:]
	    self.intercept=weights[0]
		


def main():
	try: 
		data=np.loadtxt(sys.argv[1],dtype='float',delimiter=',',usecols=(0,1,2,4))
		X=data[:,0:3]
		Y=data[:,3]
		p=Pocket(random_state=2308863)
		p.fit(X,Y)
		print "Weights:"
		print p.weights
		print "Intercept Value:"
		print p.intercept
		print "Minimum Number Of Errors:"
		print p.minErrors
		plt.plot(np.arange(0,7000),p.errorCountArr)
		plt.show()
	except Exception,e:
		print(str(e))
		print('Syntax:')
		print('\tpython perceptron.py <data_filename>')
		sys.exit()

if __name__ == "__main__":
    main()