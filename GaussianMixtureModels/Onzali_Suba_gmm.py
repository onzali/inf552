from __future__ import division
from collections import defaultdict
import pandas as pd
import numpy as np
import math
import sys
import time
from numpy import genfromtxt
import random
class GMM:
    def __init__(self,num_of_clusters=3,max_num_of_iter=200,means=[],cov_vars=[],amplitude=[],threshold=0.001):
        self.num_of_clusters=num_of_clusters
        self.max_num_of_iter=max_num_of_iter
        self.means=means
        self.cov_vars=cov_vars
        self.amplitude=amplitude
        self.threshold=threshold
        
    def fit(self,data=None):
        iter=1
        membership_weights=None
        while True:
            self.mstep(data,membership_weights)
            new_membership_weights=self.estep(data)
            if membership_weights is not None and new_membership_weights is not None:
                if (np.abs(new_membership_weights-membership_weights) < self.threshold).all():
                    break
            if iter >= self.max_num_of_iter:
                break
            iter+=1
            membership_weights=new_membership_weights
        return
    
    def kmeans(self,data):
        cluster_centroids=np.array(random.sample(data, 3))
        dict_of_clusters=defaultdict(list)
        for i in xrange(0,len(data)):
            val, min_index = min((val, idx) for (idx, val) in enumerate([self.euclidean_dist(data[i],x) for x in cluster_centroids]))
            dict_of_clusters[min_index].append(data[i].tolist())
        dict_of_clusters=[np.array(dict_of_clusters[i]) for i in dict_of_clusters]
        return dict_of_clusters
    
    def euclidean_dist(self,x,y):
        return np.sqrt(np.sum((x-y)**2))
    
    def mstep(self,data,membership_weights=None):
        self.means=[]
        self.cov_vars=[]
        self.amplitude = []
        num_of_var=len(data[0])
        num_of_data_points=len(data)
	sum=np.sum
        if membership_weights is not None:
            self.amplitude = sum(membership_weights, axis=0) / num_of_data_points
            for i in xrange(0,self.num_of_clusters):
                self.means.append(sum(np.multiply(data,membership_weights[:,i].reshape(len(data),1)),axis=0) / sum(membership_weights[:,i]))
                cov_temp_sum=np.zeros((num_of_var,num_of_var))
                for j in xrange(0,num_of_data_points):
                    temp=data[j]-self.means[i]
                    temp=np.dot(temp.T.reshape(num_of_var,1),temp.reshape(1,num_of_var))
                    temp=temp*membership_weights[j][i]
                    cov_temp_sum=np.add(cov_temp_sum,temp)
                cov_temp_sum=cov_temp_sum / sum(membership_weights[:,i])
                self.cov_vars.append(cov_temp_sum)
        else:
            clusters=self.kmeans(data)
            self.amplitude=np.ones(self.num_of_clusters) / self.num_of_clusters
            self.amplitude=self.amplitude.tolist()
            for i in xrange(0,self.num_of_clusters):
                self.means.append(np.mean(clusters[i], axis=0))
                self.cov_vars.append(np.cov(clusters[i].T))
        return 
    
    def estep(self,data):
        num_of_data_points=len(data)
        pdfs = np.empty([num_of_data_points,self.num_of_clusters])
        for i in xrange(0,self.num_of_clusters):
            m=self.means[i]
            cov=self.cov_vars[i]
            invcov=np.linalg.inv(cov)
            norm_factor = 1 / np.sqrt((2*np.pi)**2 * np.linalg.det(cov))
            for row in xrange(0,num_of_data_points):
                temp = data[row,:] - m
                temp = temp.T
                temp = np.dot(-0.5*temp, invcov)
                temp = np.dot(temp, (data[row,:] - m))
                pdfs[row][i] = norm_factor*np.exp(temp)
        membership_weights = np.empty([num_of_data_points,self.num_of_clusters])
        for i in xrange(0,num_of_data_points):
            denominator=np.sum(self.amplitude*pdfs[i])
            for j in xrange(0,self.num_of_clusters):
                membership_weights[i][j]=self.amplitude[j]*pdfs[i][j] / denominator
        return membership_weights
      

def main():
    try:
    	filename=sys.argv[1]
        
    	data = genfromtxt(filename, delimiter=',')
    	g=GMM(num_of_clusters=3)
    	g.fit(data)
    	print "Amplitudes:"
    	print g.amplitude
    	print
    	print "Means:"
    	print np.array(g.means)
    	print
    	print "Covariances:"
    	print np.array(g.cov_vars)
    except Exception,e:
        print(str(e))
        print('Syntax:')
        print('\tpython Onzali_Suba_kmeans.py <filename>')
        sys.exit() 
if __name__ == "__main__":
    main()
