'''
    file name: multi_layer_perceptron.py
    author: Keiron O'Shea (keo7@aber.ac.uk)
    description: A guided tour of the keras deep learning framework, with a simple multi layer perceptron example.
'''

import numpy as np

import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

from keras.utils import np_utils

from sklearn.metrics import confusion_matrix, classification_report

def load_data():
    '''
        keras comes with a few pre-configured dataset loaders, including:

        MNIST (http://yann.lecun.com/exdb/mnist/) - mnist.load_data()
        CIFAR10 (https://www.cs.toronto.edu/~kriz/cifar.html) - cifar10.load_data()
        CIFAR100 (https://www.cs.toronto.edu/~kriz/cifar.html) - cifar100.load_data()

        However, if you would like to use your own dataset then I'd advise you to use Pyvec:
            https://github.com/KeironO/pyvec

        In this case, this provides two different sorts of vectors.

        train/test_data provides a 2D vector representation of each MNIST digit.
        train/test_label provides the labels for each digit vector.
    '''
    (train_data, train_labels), (test_data, test_labels) = cifar10.load_data()

    # Integer containing the total number of classes found in the train_label vector.
    # So in this case, 10 digit classes.
    ll = set()
    for label in train_labels:
        ll.add(label[0])
    num_classes = len(ll)
    #num_classes = 10

    # For MLPs you need to convert the 2D image array down into a 1D array, for CNNs you don't!!
    train_data = train_data.reshape(train_data.shape[0], (train_data.shape[1]*train_data.shape[2]*train_data.shape[3])).astype("float32")/255
    test_data = test_data.reshape(test_data.shape[0], (test_data.shape[1]*test_data.shape[2]*test_data.shape[3])).astype("float32")/255

    # For multi label classification tasks, you need to convert class vectors down into binary class matrices
    train_labels = np_utils.to_categorical(train_labels, num_classes)

    return train_data, train_labels, test_data, test_labels, num_classes

def create_mlp(num_inputs, num_classes):
    '''
        Now for the important bit, the definition of the ANN architecture.

        If you are still unable to comprehend the basics of ANN architecture, think of it like this..
            -> Input Layer: Represents the data of which we aim to learn the model from,
                            in this case, a MNIST digit.
            -> Hidden Layer/s: Processes the data provided by the input layer, or prior
                               hidden layer using a activation function.
            -> Output Layer: Provides the classification score of the
                             network once trained or tested on a sample.

        This is quite a simple deep neural network that can be illustrated as:
            INPUT -> DENSE -> DENSE -> OUTPUT
    '''
    model = Sequential()

    # LAYER ONE

    model.add(Dense(1024, # Number of neurons in the layer, best keep this </= 512 per layer.
                    input_shape=(num_inputs, ) # Weirdly, keras calls the input layer here.
                                               # This is equal to the input, so in our case
                                               # height * width = input in the case of 1D images.
                    ))
    model.add(Activation("relu")) # The activation function of each neuron.
    model.add(Dropout(0.2)) # Dropout is quite useful in reducing overfitting.
                            # Here's the paper for it: https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
                            # In our case, we "drop out" 20% of the neurons after training.

    # LAYER TWO

    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    # OUTPUT LAYER

    model.add(Dense(num_classes)) # The number of neurons on the output layer is always equal to the number of classes.
    model.add(Activation("softmax")) # Softmax is used for multi-class classification, sigmoid is best used for binary tasks.

    optimiser = SGD()

    model.compile(loss="categorical_crossentropy",
                  optimizer=optimiser
                  )

    return model


def train_model(model, train_data, train_labels):
    batch_size = 128 # The batch size is incredibly important, and can determine the learning rate of the network.
                     # It's dependent on the size of your dataset, but it's best to stick to a maximum of 128.
    num_epochs = 30 # An epoch is a measure of the number of times all of the training samples are used once to update the weights.

    print ("Keep and eye on the loss, hopefully they'll both be going down!")

    model.fit(train_data, train_labels,
              batch_size=batch_size,
              nb_epoch=num_epochs,
              verbose=1, # Study the loss, and hopefully it should go down to 0.
              validation_split=0.2 # We would like to validate our training on 20% of the training dataset as a measure
                                   # of success
              )

    return model

def evaluate_model(model, test_data, test_labels, num_classes):
    # It is important to only use the test data here! We don't want to evaluate on trained data.
    # The test data should only be shown to the network once trained.
    predicted_classes = model.predict_classes(test_data, verbose=0).tolist()

    '''
        A confusion matrix is a good way of evaluating how good a model can classify.

        In simple terms, if you have a confusion matrix of...
          0 1
        0 9 1
        1 2 8

        Then it would mean that the model is capable of classifying class 0 with 90%
        accuracy, and class 1 with 85% - an averaged accuracy of 87.5%

        However, accuracy is not an adequate measure of true performance evaluation.
            Read more: http://machinelearningmastery.com/classification-accuracy-is-not-enough-more-performance-measures-you-can-use/

    '''
    print (confusion_matrix(test_labels, predicted_classes))

    '''
        For a bit more in-depth analysis, we can produce a classification report which will outline
        classification performance with a bit more weight behind it.
    '''
    print (classification_report(test_labels, predicted_classes))

if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels, num_classes = load_data()
    model = create_mlp(train_data.shape[1], num_classes)
    model = train_model(model, train_data, train_labels)
    evaluate_model(model, test_data, test_labels, num_classes)