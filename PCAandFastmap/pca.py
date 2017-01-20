import sys
import numpy as np


class PCA:

	def __init__(self,n_components=2,components_=None,explained_variance_=None):
		self.n_components=n_components
		self.components_=components_
		self.explained_variance_=explained_variance_


	def fit(self,data):
		row,col=data.shape
		means=np.mean(data,axis=0)
		data=data-means
		data_cov=np.cov(data,rowvar=0)

		eigenval,eigenvectors=np.linalg.eig(data_cov)
		self.explained_variance_=eigenval[0:self.n_components]
		eigenval_indices=np.argsort(eigenval)[::-1]
		eigenval[:]=eigenval[eigenval_indices]
		eigenvectors[:]=eigenvectors[:,eigenval_indices]
		eigenvectors_trunc=np.ones((col,self.n_components))
		self.components_ = eigenvectors[:,0:self.n_components]
		z=np.ones((row,self.n_components))
		for i in xrange(0,col):
			eigenvectors_trunc[i:]=eigenvectors[i,0:self.n_components]
		for i in xrange(0,len(data)):
			z[i]=np.dot(eigenvectors_trunc.T,data[i])


def main():
	try:
		data=np.loadtxt(sys.argv[1],dtype='float', delimiter="\t")
		k=int(sys.argv[2])
		
		pca=PCA(n_components=2)
		pca.fit(data)
		print pca.components_
	except Exception,e:
		print(str(e))
		print('Syntax:')
		print('\tpython pca.py <filename> <n_components>')
		sys.exit()



if __name__ == "__main__":
    main()