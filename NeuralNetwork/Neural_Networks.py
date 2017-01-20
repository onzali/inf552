import numpy as np
from numpy import vectorize
from scipy.special import expit as sigmoid
class NeuralNetwork:
    
    def __init__(self,inputLayerSize=960,hiddenLayerSize=100,outputLayerSize=1,learningRate=0.1,numberOfEpochs=1000):
        self.learningRate=learningRate
        self.numberOfEpochs=numberOfEpochs
        self.inputLayerSize=inputLayerSize
        self.hiddenLayerSize=hiddenLayerSize
        self.outputLayerSize=outputLayerSize
        self.W1 = np.random.uniform(low=-0.001, high=0.001, size=(self.inputLayerSize,self.hiddenLayerSize))
        self.W2 = np.random.uniform(low=-0.001, high=0.001, size=(self.hiddenLayerSize,self.outputLayerSize)) 
    
        
    def checkPredictedValue(self,z,actualZ):
        if(z==actualZ):
            return True
        else:
            return False
        
    def fit(self,x,y):

        for i in range(self.numberOfEpochs):
    
            # Dataset loop
            for j in range(x.shape[0]):

                self.yHat=self.propagate(x[j])
                self.calculateDeltas(x[j],y[j])
        
    def propagate(self,x):

        self.z2=np.dot(x, self.W1)

        # Hidden layer activation
        self.a2 = self.sig(self.z2)
            
        # Adding bias to the hidden layer
        # ah = np.concatenate((np.ones(1).T, np.array(ah))) 

        self.z3=np.dot(self.a2, self.W2)

        # Output activation
        return self.sig(self.z3)
    
    
    def sig(self,x):
        return 1.0/(1.0 + np.exp(-x))

    def sigmoid_prime(self,x):
        return self.sig(x)*(1.0-self.sig(x))
    
    def tanh(self,x):
        return np.tanh(x)

    def tanh_prime(self,x):
        return 1.0 - np.tanh(x)**2
    
    def calculateDeltas(self,x,y):
        yH=self.yHat
        learningRate=self.learningRate
        sigmoid_prime=self.sigmoid_prime

        error=y - yH
        # Deltas    
        delta_output = np.multiply(error,sigmoid_prime(yH))
        delta_hidden = np.multiply(np.dot(self.W2, delta_output),sigmoid_prime(self.a2))

        de_oweight=np.outer(self.a2,delta_output)
        de_hweight=np.outer(x,delta_hidden[0:])

        self.W2 += np.multiply(learningRate, de_oweight)
        self.W1 += np.multiply(learningRate, de_hweight)

    def predict(self, x): 

        # Allocate memory for the outputs
        y = np.zeros([x.shape[0],self.W2.shape[1]])

        # Loop the inputs
        for i in range(0,x.shape[0]):

            # Outputs
            y[i] = self.propagate(x[i])

        # Return the results
        return y

def main():
    training_images = []
    training_labels = []

    with open('downgesture_train.list') as f:
        for training_image in f.readlines():
            training_image = training_image.strip()
            training_images.append(load_pgm_image(training_image))
            if 'down' in training_image:
                training_labels.append(1)
            else:
                training_labels.append(0)
    x = np.array(training_images)
    y = np.array(training_labels)
    nn=NeuralNetwork()
    nn.fit(x,y)
    
    total = 0
    correct = 0
    with open('downgesture_test.list') as f:
        total += 1
        for test_image in f.readlines():
            test_image = test_image.strip()
            p = nn.predict(np.array([load_pgm_image(test_image),]))
            p=0 if p < 0.5 else 1
            print('{}: {}'.format(test_image,p))
            if (p != 0) == ('down' in test_image):
                correct += 1

    print('correct rate: {}'.format(correct / total))

def load_pgm_image(pgm):
    with open(pgm, 'rb') as f:
        f.readline()   # skip P5
        f.readline()   # skip the comment line
        xs, ys = f.readline().split()  # size of the image
        xs = int(xs)
        ys = int(ys)
        max_scale = int(f.readline().strip())

        image = []
        for _ in range(xs * ys):
            image.append(f.read(1)[0] / max_scale)

        return image

def checkPredictedValue(z,actualZ):
    if z==actualZ:
        return True
    else:
        return False
                    
if __name__ == "__main__":
    main()
