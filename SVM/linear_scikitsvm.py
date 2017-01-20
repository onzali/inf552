from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
def main():
	data=np.loadtxt('linsep.txt',dtype='float',delimiter=',')
	X=data[:,0:2]
	Y=data[:,2]
	clf = svm.SVC(kernel='linear')
	clf.fit(X, Y)  
	print("Intercept:")
	print(clf.intercept_)
	print("Weights:")
	print(clf.coef_[0])
	plt.scatter(X[:,0],X[:,1],c=Y,cmap='bwr',alpha=1,s=50,edgecolors='k')
	x2_lefttargeth = -(clf.coef_[0][0]*(-1)+clf.intercept_)/clf.coef_[0][1]
	x2_righttargeth = -(clf.coef_[0][0]*(1)+clf.intercept_)/clf.coef_[0][1]
	plt.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],facecolors='none',s=100, edgecolors='k')
	plt.plot([-1,1], [x2_lefttargeth,x2_righttargeth])
	plt.show()	

if __name__ == "__main__":
	main()
