from numpy import vectorize
import sys
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
		
def main():
	try:
		data=np.loadtxt(sys.argv[1],dtype='float',delimiter=',',usecols=(0,1,2))
		X=data[:,0:2]
		Y=data[:,2]
		p=LinearRegression()
		p.fit(X,Y)
		print "Weights:"
		print p.coef_
		print "Intercept"
		print p.intercept_
	except Exception,e:
		print(str(e))
		print('Syntax:')
		print('\tpython sklearn_linearRegression.py <data_filename>')
		sys.exit()


if __name__ == "__main__":
    main()