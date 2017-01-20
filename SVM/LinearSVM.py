import numpy as np
import cvxopt
import matplotlib.pyplot as plt

class LinearSVM:

	def fit(self, X, Y):
		rows, cols = X.shape
		
		Q = np.zeros((rows, rows))
		for i in range(rows):
			for j in range(rows):
				Q[i,j] = np.dot(X[i], X[j])
		P = cvxopt.matrix(np.outer(Y,Y) * Q)
		q = cvxopt.matrix(np.ones(rows) * -1)
		A = cvxopt.matrix(Y, (1,rows))
		b = cvxopt.matrix(0.0)

		G = cvxopt.matrix(np.diag(np.ones(rows) * -1))
		h = cvxopt.matrix(np.zeros(rows))

		alphas = np.array(cvxopt.solvers.qp(P, q, G, h, A, b)['x']).reshape(1,rows)[0]

		support_vector_indices = np.where(alphas>0.00001)[0]

		self.alphas = alphas[support_vector_indices]
		self.support_vectors = X[support_vector_indices]
		self.support_vectors_y = Y[support_vector_indices]
		print("%d support vectors out of %d points" % (len(self.alphas), rows))

		self.weights = np.zeros(cols)
		for i in range(len(self.alphas)):
			self.weights += self.alphas[i] * self.support_vectors_y[i] * self.support_vectors[i]

		self.intercept = self.support_vectors_y[0] - np.dot(self.weights, self.support_vectors[0])


def main():
	data=np.loadtxt('linsep.txt',dtype='float',delimiter=',')
	X=data[:,0:2]
	Y=data[:,2]
	p=LinearSVM()
	p.fit(X,Y)
	print("Intercept:")
	print(p.intercept)
	print("Weights:")
	print(p.weights)
	plt.scatter(X[:,0],X[:,1],c=Y,cmap='bwr',alpha=1,s=50,edgecolors='k')
	x2_lefttargeth = -(p.weights[0]*(-1)+p.intercept)/p.weights[1]
	x2_righttargeth = -(p.weights[0]*(1)+p.intercept)/p.weights[1]
	plt.scatter(p.support_vectors[:,0],p.support_vectors[:,1],facecolors='none',s=100, edgecolors='k')
	plt.plot([-1,1], [x2_lefttargeth,x2_righttargeth])
	plt.show()	

if __name__ == "__main__":
	main()
