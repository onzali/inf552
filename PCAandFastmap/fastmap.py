from __future__ import division
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
		
class FastMap:

	def __init__(self, n_components=2, results=None):
		self.n_components=n_components
		self.results=results

	def fastmap(self,data,list_obj,obj_A,obj_B,max_distance):
		row,col=data.shape
		d_ab=max_distance
		x=[]
		for j in xrange(0,len(list_obj)):
			d_a=0
			d_b=d_ab
			
			for i in xrange(0,len(data)):

				if(((data[i][1]==obj_A and data[i][0]==list_obj[j]) or (data[i][1]==list_obj[j] and data[i][0]==obj_A)) and list_obj[j]!=obj_A):

					d_a=data[i][2]
					
				elif(((data[i][1]==obj_B and data[i][0]==list_obj[j]) or (data[i][1]==list_obj[j] and data[i][0]==obj_B)) and list_obj[j]!=obj_B): 
					
					d_b=data[i][2]
					
			
			if(list_obj[j]!=obj_A and list_obj[j]!=obj_B):
				x.append((np.square(d_a)+np.square(d_ab)-np.square(d_b)) / (2*d_ab))
			elif(list_obj[j]==obj_A):
				x.append(d_a)
			elif(list_obj[j]==obj_B):
				x.append(d_b)
		return x

	def findMaxDistance(self,obj_list,data):

		obj=random.choice(obj_list)
		next_obj=-1
		max_distance=0
		iter=0
		while True:
			max_distance=0
			for i in xrange(0,len(data)):
				if data[i][0]==obj or data[i][1]==obj:
					if data[i][2]>max_distance:
						max_distance=data[i][2]
						if data[i][0]==obj:
							next_obj=data[i][1]
						else:
							next_obj=data[i][0]
			if iter==5:
				break
			iter=iter+1
			obj=next_obj
		return obj,next_obj,max_distance


		

	def fit(self,data):
		
		results=[]
		for i in xrange(0,self.n_components):
			list_obj=[]
			for i in xrange(0,len(data)):
				for j in xrange(0,2):
					if(data[i][j] not in list_obj):
						list_obj.append(data[i][j])
			objA,objB,max_distance=self.findMaxDistance(list_obj,data)			
			x=self.fastmap(data,list_obj,objA,objB,max_distance)
			
			results.append(x)
			
			for j in xrange(0,len(data)):
				data[j][2]=(np.sqrt((np.square(data[j][2])-np.square(x[int(data[j][0])-1]-x[int(data[j][1])-1]))))
		self.results=np.array(results).T
	

def main():
	try: 
		data=np.loadtxt(sys.argv[1],dtype='float', delimiter="\t")
		data_point_label=np.loadtxt(sys.argv[2],dtype='string')
		print data_point_label
		k=int(sys.argv[3])
		fm=FastMap(n_components=2)
		fm.fit(data)
		print fm.results	
		fig, ax = plt.subplots()
		ax.scatter(fm.results[:,0],fm.results[:,1])

		for i, txt in enumerate(data_point_label):
		    ax.annotate(txt, (fm.results[i,0],fm.results[i,1]))
		plt.show()
		
	except Exception,e:
		print(str(e))
		print('Syntax:')
		print('\tpython fastmap.py <data_filename> <wordlist_filename> <n_components>')
		sys.exit()

		

		


if __name__ == "__main__":
    main()
