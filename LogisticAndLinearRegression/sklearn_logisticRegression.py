from numpy import vectorize
import sys
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
		
def main():
	try:
		data=np.loadtxt(sys.argv[1],dtype='float',delimiter=',',usecols=(0,1,2,4))
		X=data[:,0:3]
		Y=data[:,3]
		p=LogisticRegression(max_iter=7000)
		p.fit(X,Y)
		print "Weights:"
		print p.coef_
		print "Intercept"
		print p.intercept_
	except Exception,e:
		print(str(e))
		print('Syntax:')
		print('\tpython sklearn_logisticRegression.py <data_filename>')
		sys.exit()


if __name__ == "__main__":
    main()