import cv2
import glob
import random
import numpy as np
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Emotion list
#fishface = cv2.createFisherFaceRecognizer() #Initialize fisher face classifier

data = {}

def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("dataset/%s/*" %emotion)
    random.seed(1)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction

def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training, prediction = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            training_data.append(gray) #append image array to training data list
            training_labels.append(emotions.index(emotion))
    
        for item in prediction: #repeat above process for prediction set
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels

training_data, training_labels, prediction_data, prediction_labels = make_sets()

xtrain=np.zeros((len(training_data),122500))
for i in xrange(0,len(training_data)):
    xtrain[i]=np.ravel(training_data[i])

ytrain=np.array(training_labels)

xtest=np.zeros((len(prediction_data),122500))
for i in xrange(0,len(prediction_data)):
    xtest[i]=np.ravel(prediction_data[i])

ytest=np.array(prediction_labels)

lda = LinearDiscriminantAnalysis()

lda.fit(xtrain,ytrain)

x_train_lda=lda.transform(xtrain)
x_test_lda=lda.transform(xtest)


results = []
clf = SVC(kernel='linear',random_state=1)

clf.fit(x_train_lda,ytrain)

predicted=clf.predict(x_test_lda)

print("Classification report:")
print(metrics.classification_report(ytest, predicted))
print("Confusion Matrix")
print( metrics.confusion_matrix(ytest, predicted))
print("\nPrediction Score")
print(clf.score(x_test_lda, ytest))
