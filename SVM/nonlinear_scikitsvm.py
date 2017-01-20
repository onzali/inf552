from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

def main():
	data=np.loadtxt('nonlinsep.txt',dtype='float',delimiter=',')
	X=data[:,0:2]
	Y=data[:,2]
	clf = svm.SVC(kernel='poly',degree=2)
	clf.fit(X, Y)  

	print("Intercept:")
	print(clf.intercept_)
	print("Weights:")
	print(clf.dual_coef_[0])
	

if __name__ == "__main__":
	main()
