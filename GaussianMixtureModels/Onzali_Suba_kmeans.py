from collections import defaultdict
import numpy as np
import sys
from numpy import genfromtxt
import random

class KMeans:
    def __init__(self,num_of_clusters=3,cluster_centroids=None,max_num_of_iter=50,threshold=0.001):
        self.num_of_clusters=num_of_clusters
        self.cluster_centroids=cluster_centroids
        self.max_num_of_iter=max_num_of_iter
	self.threshold=threshold
        
    def fit(self,data=None):
        iter=1
        cluster_centroids=np.array(random.sample(data, 3))
        while True:
            new_cluster_centroids=np.array(self.compute_centroids(data,cluster_centroids))
            if np.array_equal(cluster_centroids,new_cluster_centroids):
                break
            elif (np.abs(new_cluster_centroids-cluster_centroids) < self.threshold).all():
                break
	    if iter >= self.max_num_of_iter:
                break
            iter+=1
            cluster_centroids=new_cluster_centroids
        return new_cluster_centroids
        
    def euclidean_dist(self,x,y):   
        return np.sqrt(np.sum((x-y)**2))
        
    def compute_centroids(self,data,cluster_centroids):
        dict_of_clusters=defaultdict(list)
        dict_of_clusters_indices=defaultdict(list)
        for i in xrange(0,len(data)):
            val, min_index = min((val, idx) for (idx, val) in enumerate([self.euclidean_dist(data[i],x) for x in cluster_centroids]))
            dict_of_clusters_indices[min_index].append(i)
            dict_of_clusters[min_index].append(data[i].tolist())

        new_cluster_centroids=[]
        for cluster in dict_of_clusters:
            new_cluster_centroids.append( np.array(dict_of_clusters[cluster]).mean(axis=0))

        return new_cluster_centroids

def main():
    try:
        filename='clusters.txt'
        
        data = genfromtxt(filename, delimiter=',')
        clf=KMeans()
        centroids=clf.fit(data)
	print "Cluster Centroids:"
        print centroids

    except Exception,e:
        print(str(e))
        print('Syntax:')
        print('\tpython Onzali_Suba_kmeans.py <filename>')
        sys.exit()

if __name__ == "__main__":
    main()
