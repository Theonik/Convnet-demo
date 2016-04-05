import numpy as np
import argparse
import os
import utils
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD

from keras.utils import np_utils

from sklearn.metrics import confusion_matrix, classification_report


def load_data():
    (train_data, train_labels), (test_data, test_labels) = cifar10.load_data()

    ll = set()
    for label in train_labels:
        ll.add(label[0])
    num_classes = len(ll)

    if greyscale:
        train_data = utils.convert_set_to_greyscale(train_data, 2)
        test_data =  utils.convert_set_to_greyscale(test_data, 2)
    else:
        train_data = train_data.astype('float32') / 255
        test_data = test_data.astype('float32') / 255

    val_data = None
    val_labels = None

    if augment_data: # Split data for validation if data augmentation is used
        rng_state = np.random.get_state()
        np.random.shuffle(train_data)
        np.random.set_state(rng_state)
        np.random.shuffle(train_labels)
        split_0 = int(train_data.shape[0]*0.2)
        split_1 = train_data.shape[0] - split_0
        train_data, foo, val_data = np.split(train_data, [split_1, split_0])
        train_labels, foo, val_labels = np.split(train_labels, [split_1, split_0])
        val_labels = np_utils.to_categorical(val_labels, num_classes)

    train_labels = np_utils.to_categorical(train_labels, num_classes)
    #test_labels = np_utils.to_categorical(test_labels, num_classes)

    return train_data, train_labels, val_data, val_labels, test_data, test_labels, num_classes


def create_cnn(channels, rows, columns, num_classes):

    model = Sequential()

    # LAYER ONE

    model.add(Convolution2D(8, 3, 3,
                            input_shape=(channels, rows, columns)
                            )
              )
    model.add(Activation('relu'))
    model.add(Convolution2D(8, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid', dim_ordering='th'))
    model.add(Dropout(0.2))

    # LAYER TWO

    model.add(Convolution2D(14, 3, 3))
    model.add(Activation('relu'))
    model.add(Convolution2D(14, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid', dim_ordering='th'))
    model.add(Dropout(0.2))

    # OUTPUT LAYER
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Dropout(0.4))
    model.add(Dense(256))
    model.add(Dropout(0.4))
    model.add(Activation('relu'))
    model.add(Dense(num_classes)) # The number of neurons on the output layer is always equal to the number of classes.
    model.add(Activation("softmax")) # Softmax is used for multi-class classification, sigmoid is best used for binary tasks.

    optimiser = SGD(lr=0.01, momentum=0.90, decay=1e-8, nesterov=True)

    model.compile(loss="categorical_crossentropy",
                  optimizer=optimiser
                  )

    return model


def train_model(model, train_data, train_labels, val_data, val_labels):
    batch_size = 128
    num_epochs = 1000
    print ("If the model does not overfit both loss numbers should go down down down")
    early_stop = EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
    if augment_data:
        data_aug = ImageDataGenerator(
            featurewise_center=False, # set input mean to 0 over the dataset
            samplewise_center=False, # set each sample mean to 0
            featurewise_std_normalization=False, # divide inputs by std of the dataset
            samplewise_std_normalization=False, # divide each input by its std
            zca_whitening=False,
            rotation_range=0.1,
            width_shift_range=0.08,
            height_shift_range=0.08,
            horizontal_flip=True,
            vertical_flip=False,
        )
        data_aug.fit(train_data)
        model.fit_generator(data_aug.flow(train_data, train_labels, batch_size=batch_size, shuffle=True),
                            samples_per_epoch=train_data.shape[0],
                            nb_epoch=num_epochs,
                            verbose=1,
                            validation_data=(val_data, val_labels),
                            callbacks=[early_stop]
                            )
    else:
        model.fit(train_data, train_labels,
                  batch_size=batch_size,
                  nb_epoch=num_epochs,
                  verbose=1,
                  shuffle=True,
                  validation_split=0.20,
                  callbacks=[early_stop]
                  )

    return model


def evaluate_model(model, test_data, test_labels, num_classes):

    predicted_classes = model.predict_classes(test_data, verbose=0).tolist()

    print (confusion_matrix(test_labels, predicted_classes))

    print (classification_report(test_labels, predicted_classes))

def save_model(model, path='./models/', name='model'):
    json_string = model.to_json()
    open(path + name + '.json', 'w').write(json_string)
    model.save_weights(path + name + '_weights.h5')

def export_model(model, path='./models/', name='model'):
    json_string = model.to_json()
    open(path + name + '.json', 'w').write(json_string)

def load_model(path):
    l_model = model_from_json(open(path + '.json').read())
    l_model.load_weights(path + '_weights.h5')
    return l_model

def load_model(path):
    l_model = model_from_json(open(path + '.json').read())
    return l_model

#def parse_input():


if __name__ == "__main__":
    augment_data = True
    greyscale = True
    train_data, train_labels, val_data, val_labels, test_data, test_labels, num_classes = load_data()
    model = create_cnn(train_data.shape[1], train_data.shape[2], train_data.shape[3], num_classes)
    model = train_model(model, train_data, train_labels, val_data, val_labels)
    evaluate_model(model, test_data, test_labels, num_classes)
