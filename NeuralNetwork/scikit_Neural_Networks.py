#!/usr/bin/python3

# This is a script using sklearn to do the homework.

from sklearn.neural_network import MLPClassifier

def load_pgm_image(pgm):
    with open(pgm, 'rb') as f:
        f.readline()   # skip P5
        f.readline()   # skip the comment line
        xs, ys = f.readline().split()  # size of the image
        xs = int(xs)
        ys = int(ys)
        
        max_scale = int(f.readline().strip())
        
        image = []
        for i in range(xs * ys):
            image.append((f.read(1)[0]) / max_scale)
    

        return image

images = []
labels = []

with open('downgesture_train.list') as f:
    for training_image in f.readlines():
        
        training_image = training_image.strip()
        images.append(load_pgm_image(training_image))
        if 'down' in training_image:
            labels.append(1)
        else:
            labels.append(0)

c = MLPClassifier(solver='sgd', alpha=0,
                  hidden_layer_sizes=(100,), activation='logistic', learning_rate_init=0.1,
                  max_iter=1000)

c.fit(images, labels)

total = 0
correct = 0
with open('downgesture_test.list') as f:
    total += 1
    for test_image in f.readlines():
        test_image = test_image.strip()
        p = c.predict([load_pgm_image(test_image),])[0]
        print('{}: {}'.format(test_image, p))
        if (p != 0) == ('down' in test_image):
            correct += 1
print('correct rate: {}'.format(correct / total))
