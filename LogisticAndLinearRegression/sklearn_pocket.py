from numpy import vectorize
import sys
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
		
def main():
	try:
		data=np.loadtxt('classification.txt',dtype='float',delimiter=',',usecols=(0,1,2,4))
		X=data[:,0:3]
		Y=data[:,3]
		p=Perceptron(random_state=23088,n_iter=7000)
		p.fit(X,Y)
		print "Weights:"
		print p.coef_
		print "Intercept"
		print p.intercept_
	except Exception,e:
		print(str(e))
		print('Syntax:')
		print('\tpython sklearn_pocket.py <data_filename>')
		sys.exit()


if __name__ == "__main__":
    main()