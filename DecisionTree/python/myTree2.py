from __future__ import division
from collections import defaultdict
import pandas as pd
import numpy as np
import math
import sys
import time



class DecisionTreeClassifier:
    
    def __init__(self, rootNode=None,features=None, targetAttr=None, height=1):
        self.rootNode=rootNode
        self.height=height
        self.targetAttr=targetAttr
        self.features=features

    class DecisionNode:

        def __init__(self, attr, branches=defaultdict(), outcome=defaultdict(list)):
            self.attr=attr
            self.branches=branches
            self.outcome=outcome

    def entropy(self,labelCount):
        en=0
        for count in labelCount:
            en-=count / sum(labelCount)*math.log(count / sum(labelCount))
        return en

    def train(self,df=None,maxNumOfInstances=1,level=0,attrVal=None):

        maxInfoGain=0
        maxAttr=None
        maxDfs=defaultdict(None)
        branchNodes=defaultdict(None)

        parentClassLabelsSize=df.groupby([self.targetAttr]).size() #get the number of instances for each class label

        if len(parentClassLabelsSize)<2: #if pure node
            return self.DecisionNode(self.targetAttr, None,outcome=parentClassLabelsSize.to_dict())

        if len(df)<=maxNumOfInstances:  #if number of instances go below a limit 
             return self.DecisionNode(self.targetAttr, None,outcome=parentClassLabelsSize.to_dict())

        parentEntropy=self.entropy(parentClassLabelsSize) #find the parent entropy 
        
        for feature in self.features:
            uniqueFeatureValues=df[feature].unique()
            en=0
            dfs=defaultdict(None) #list of dataframe for each child node
            for val in uniqueFeatureValues:
                childDF=df[df[feature]==val]
                childDFSize=childDF.groupby(self.targetAttr).size()

                if len(childDFSize)>1: #if more than one labels in the new Dataframe 
                    en+=(childDFSize.sum() / parentClassLabelsSize.sum())*self.entropy(childDFSize) #weighted avg of child entropies
                dfs[val]=childDF

            infoGain=parentEntropy-en #calculating information gain

            if maxInfoGain<infoGain: #if current information gain is higher
                maxInfoGain=infoGain
                maxAttr=feature
                maxDfs=dfs

        for attrValue in maxDfs:
            branchNodes[attrValue]=self.train(maxDfs[attrValue],maxNumOfInstances=maxNumOfInstances,level=level+1,attrVal=attrValue)

        return self.DecisionNode(maxAttr,branchNodes)

    def buildTree(self,data=None,target=None,maxNumOfInstances=1):

        df=pd.concat([data, target], axis=1, join='inner') #create a dataframe combining the training data and training label
        columns=df.columns
        colLength=len(columns)
        self.features=columns[0:colLength-1]
        self.targetAttr=columns[colLength-1]
        self.rootNode=self.train(df,maxNumOfInstances,level=0,attrVal=None)
        self.height=self.heightOfTree(self.rootNode)

    def predict(self,testData):

        testList=[]
        for i in xrange(0,len(testData)):
            res=testData[:][i:i+1]
            outcome=self.getBranch(self.rootNode,self.rootNode.attr,res,rowNumber=i)
            if len(outcome)>1:
                newOutcome=dict()
                for k in outcome:
                    prob=outcome[k] / sum(outcome.values())
                    newOutcome[k]=round(prob,2)
                testList.append(str(newOutcome))
            else:
                testList.append(str(outcome.keys()[0]))
        testOutcome=pd.Series(testList)
        testData[self.targetAttr]=testOutcome
        return testData

    def getBranch(self, node, attr, testData, rowNumber=0):
        if node.branches==None:
            return node.outcome

        testDataAttrVal=testData[attr][rowNumber]
        nextNode=node.branches[testDataAttrVal]
        return self.getBranch(nextNode,nextNode.attr,testData, rowNumber)

    def drawTree(self,node=None,level=0):
        if node==None:
            node=self.rootNode
        for i in xrange(1,self.height+1):
            if node!=None:
                p=self.printLevel(node,i)
                if p!=None:
                    print(''+str(p)+''),
                    print
            
    def printLevel(self,node,level):
        if node==None:
            return ''
        if level==1:
            if len(node.outcome)!=0:
                return '->'+node.attr+' '+str(node.outcome)
            else:
                return '->'+node.attr
        if level==1 and node.branches==None:
            return '->'+node.attr
        else:
            res='['
            if node.branches!=None:
                for b in node.branches:
                    if b!=None:
                        n=self.printLevel(node.branches[b],level-1)
                        if level==2:
                            res+=str(b)
                        res+=str(n)+'\t'
                        
            return res+']'
        
    def heightOfTree(self,node=None):  #determines the height of the tree
        if node==None or node.branches==None:
            return 1
        else:
            if node.branches!=None:
                h=1+max(self.heightOfTree(node.branches[b]) for b in node.branches)
                return h
                
        
                    
                        
