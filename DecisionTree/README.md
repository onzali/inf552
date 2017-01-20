# inf552

To use the Decision Tree implemented - python runMyTree.py <trainingFile> <testFile>

To use the Decision Tree implemented in scikit learn - python sklearn_decisiontree.py <trainingFile> <testFile>

Step 1: Modified the input file to remove the circular brackets from the header and the row indicators from each line.

Step 2: Used a pandas dataframe to read the training and test files since it provides additional functionalities like groupBy

Scikit learn – Machine Learning Library in Python 

Class sklearn.tree.DecisionTreeClassifier is used for classifying datasets into various class labels.
•	Builds a binary tree
•	Can use gini index or entropy as a criterion to find the best split
•	Provides multiple termination criteria
•	Treats 0 as True and 1 as False
•	Drawback - does not take categorical features directly. The categorical data needs to be converted to a numerical representation of 	zeros and ones (One-hot encoding) and then fed to the DecisionTreeClassifier.

	There are many ways one can convert their categorical data into a vector of 1’s and 0’s:

	1.	Pandas library has a get_dummies() - function that creates a column for each unique value in the feature and represents that unique value in the that column with ‘1.0’ and the remaining with ‘0.0’

		import pandas as pd

		df=pd.read_csv(filename)
		features=df.columns[0:len(df.columns)-1]
		trainingData=df.iloc[:,:len(features)]

		for col in features:
			dummies = pd.get_dummies(data[col],drop_first).rename(columns=lambda x: col+'_' + str(x))
			data=pd.concat([data, dummies], axis=1)
			data=data.drop([col], axis=1)
			
	2.	Scikit learn’s DictVectorizer  - takes in a list of rows represented by a dictionary. If trainingData represents the input data, then trainingData.T.to_dict().values() can be used as an input the DictVectorizer  

		from sklearn.feature_extraction import DictVectorizer
		
		vec = DictVectorizer()
		vec.fit_transform(trainingData.T.to_dict().values()).toarray()

	3.	OneHotEncoder along with LabelEncoder – LabelEncoder converts the categorical text/image features into a categorical numerical features. The resulting categorical numerical features then is used to obtain a one-hot encoding for the intial dataset.

		from sklearn import preprocessing
		
		res=pd.DataFrame()
		le = preprocessing.LabelEncoder()
		for col in features:
    		le.fit(trainingData[col])
    		res[col]=pd.Series(le.transform(trainingData[col]))


Improvements :
	•	Currently the decision tree we have implemented takes 110 ms to build the tree compared to 0.6 ms taken by the scikit learn DecisionTreeClassifier. Similarly the time taken for our Classifier to predict is around 5 ms and that taken by scikit learn is 0.4 ms. Better structures might help improve the performance of the Decision Tree
	•	Can include more termination criteria to avoid large trees and overfitting
	•	We can implement the same decision tree in the form of a binary decision tree to avoid wide trees and better representation.


