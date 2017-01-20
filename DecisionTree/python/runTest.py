from __future__ import division
from collections import defaultdict
import myTree2 as m
import math
import sys
import pandas as pd
from sklearn import tree
import time

def main():
    try:
        filename=sys.argv[1]
        testFilename=sys.argv[2]

        df=pd.read_csv(filename)
        data=df.iloc[:,:len(df.columns)-1]
        target=df.iloc[:,len(df.columns)-1:]

        clf=m.DecisionTreeClassifier()
       
        clf.buildTree(data,target)
       
        clf.drawTree()
        print 

        testData=pd.read_csv(testFilename)
        print(clf.predict(testData))
    except Exception,e:
        print(str(e))
        sys.exit()
    
if __name__ == "__main__":
    main()
