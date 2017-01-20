from sklearn import tree
import sys
import pandas as pd
import numpy as np
import time

def featureExpansion(data,col):
	dummies = pd.get_dummies(data[col]).rename(columns=lambda x: col+'_' + str(x))
	data=pd.concat([data, dummies], axis=1)
	data=data.drop([col], axis=1)
	return data

def main():
    try:
        trainingFile=sys.argv[1]
        testFile=sys.argv[2]

       	df=pd.read_csv(trainingFile)
       	testData=pd.read_csv(testFile)

       	columns=df.columns
       	trainingData=df.iloc[:,:len(columns)-1]
       	trainingLabels=df.iloc[:,len(columns)-1:]
       	features=columns[:len(columns)-1]
       	targetLabel=columns[len(columns)-1:]

       	#using pandas get_dummies function convert training features
       	for col in features:
            trainingData=featureExpansion(trainingData,col)

        #using pandas get_dummies function cnovert target variable
       	trainingLabels=featureExpansion(trainingLabels,targetLabel[0])

       	clf = tree.DecisionTreeClassifier(criterion='entropy')
       	
       	clf = clf.fit(trainingData, trainingLabels)
       	
       	tree.export_graphviz(clf,out_file='sklearn_tree.dot')

       	originalData=testData
       	#using pandas get_dummies function convert test features
       	for col in features:
           	testData=featureExpansion(testData,col)

       	#include the missing features columns in the test data
       	addCols=list(set(trainingData.columns)-set(testData.columns))
       	newDF=pd.DataFrame(0.0,index=np.arange(len(testData)),columns=addCols)
       	testData=pd.concat([testData,newDF], axis=1)

        res=pd.DataFrame(clf.predict(testData),columns=trainingLabels.columns)

       	testLabel=pd.DataFrame()
       	for col in res.columns:
       		val=col.split('_')[1]
    		res[res[col]==0]=val

    	print(pd.concat([originalData,pd.DataFrame(res[res.columns[0]])],axis=1).rename(columns = {res.columns[0]:targetLabel[0]}))

    except Exception,e:
        print(str(e))
        sys.exit()
    
if __name__ == "__main__":
    main()
