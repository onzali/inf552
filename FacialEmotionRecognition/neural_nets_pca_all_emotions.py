import cv2
import glob
import random
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
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

x=np.zeros((len(training_data),122500))
for i in xrange(0,len(training_data)):
    x[i]=np.ravel(training_data[i])

y=np.array(training_labels)

xtest=np.zeros((len(prediction_data),122500))
for i in xrange(0,len(prediction_data)):
    xtest[i]=np.ravel(prediction_data[i])

ytest=np.array(prediction_labels)

pca = PCA(n_components=100, svd_solver='randomized', whiten=True).fit(x)

X_train_pca = pca.transform(x)

X_test_pca = pca.transform(xtest)

clf=MLPClassifier(activation='relu',hidden_layer_sizes=(1000,1000,1000),solver="sgd",random_state=1)

clf.fit(X_train_pca,y)

predicted=clf.predict(X_test_pca)

print("Classification report:")
print(metrics.classification_report(ytest, predicted))
print("Confusion Matrix")
print( metrics.confusion_matrix(ytest, predicted))
print("\nPrediction Score")
print(clf.score(X_test_pca, ytest))
