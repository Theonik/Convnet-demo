import numpy as np
import matplotlib.pyplot as plot
# import cv2 as ocv
# import argparse
import os
import utils
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
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
        split_0 = int(train_data.shape[0]*valid_split)
        split_1 = train_data.shape[0] - split_0
        train_data, foo, val_data = np.split(train_data, [split_1, split_0])
        train_labels, foo, val_labels = np.split(train_labels, [split_1, split_0])
        val_labels = np_utils.to_categorical(val_labels, num_classes)

    train_labels = np_utils.to_categorical(train_labels, num_classes)

    return train_data, train_labels, val_data, val_labels, test_data, test_labels, num_classes


def create_cnn(channels, rows, columns, num_classes):

    model = Sequential()

    # LAYER ONE

    model.add(Convolution2D(48, 3, 3,
                            input_shape=(channels, rows, columns)
                            )
              )
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='th'))
    model.add(Dropout(0.2))

    # LAYER TWO

    model.add(Convolution2D(96, 3, 3))
    model.add(Activation('relu'))
    model.add(Convolution2D(96, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='th'))
    model.add(Dropout(0.2))

    # OUTPUT LAYER
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes)) # The number of neurons on the output layer is always equal to the number of classes.
    model.add(Activation("softmax")) # Softmax is used for multi-class classification, sigmoid is best used for binary tasks.

    optimiser = SGD(lr=0.01, momentum=0.90, decay=1e-6, nesterov=True)

    model.compile(loss="categorical_crossentropy",
                  optimizer=optimiser
                  )

    return model


def train_model(model, train_data, train_labels, val_data, val_labels):
    batch_size = 128
    num_epochs = 5000
    print ("If the model does not overfit both loss numbers should go down down down")
    early_stop = EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
    checkpoint = ModelCheckpoint(check_path,monitor='val_loss', verbose=0, save_best_only=True)
    if augment_data:
        data_aug = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,
            rotation_range=0.1,
            width_shift_range=0.08,
            height_shift_range=0.08,
            horizontal_flip=True,
            vertical_flip=False,
        )
        data_aug.fit(train_data)
        hist = model.fit_generator(data_aug.flow(train_data, train_labels, batch_size=batch_size, shuffle=True),
                                   samples_per_epoch=train_data.shape[0],
                                   nb_epoch=num_epochs,
                                   verbose=1,
                                   validation_data=(val_data, val_labels),
                                   callbacks=[early_stop, checkpoint]
                                   )
    else:
        hist = model.fit(train_data, train_labels,
                         batch_size=batch_size,
                         nb_epoch=num_epochs,
                         verbose=1,
                         shuffle=True,
                         validation_split=valid_split,
                         callbacks=[early_stop, checkpoint]
                         )

    return model, hist


def evaluate_model(model, hist, test_data, test_labels, path='./models', name='network'):

    predicted_classes = model.predict_classes(test_data, verbose=0).tolist()
    cm = confusion_matrix(test_labels, predicted_classes)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cr = classification_report(test_labels, predicted_classes)

    plot_confusion_matrix(cm, lookup_labels, path + '/' + name)
    plot_confusion_matrix(cm_normalized, lookup_labels, path + '/' + name + '_normalized')  # Save Normalised Confusion plot
    plot_learning_rate(hist, path + '/' + name)  # Save history plot
    # save reports
    f = open(path + '/' + name + '_report.txt', 'w')
    f.write(cr)
    f.close()
    f = open(path + '/' + name + '_matrix.txt', 'w')
    f.write(str(cm))
    f.close()
    f = open(path + '/' + name + '_normal_matrix.txt', 'w')
    f.write(str(cm_normalized))
    f.close()

    print (cm)

    print (cr)


def generate_reports(path, test_data, test_labels):
    for model_file in os.listdir(path):
        if model_file.endswith('.json'):
            print 'loading report for ' + model_file
            eval_model = load_model(path + '/' + os.path.splitext(model_file)[0])
            predicted_classes = eval_model.predict_classes(test_data, verbose=0).tolist()
            cm = confusion_matrix(test_labels, predicted_classes)
            plot_confusion_matrix(cm, lookup_labels, path + '/' + os.path.splitext(model_file)[0])
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            plot_confusion_matrix(cm_normalized, lookup_labels, path + '/' + os.path.splitext(model_file)[0] + '_normalized')
            f = open(path + '/' + os.path.splitext(model_file)[0] + '_report.txt', 'w')
            f.write(classification_report(test_labels, predicted_classes))
            f.close()
            f = open(path + '/' + os.path.splitext(model_file)[0] + '_matrix.txt', 'w')
            f.write(str(cm))
            f.close()
            f = open(path + '/' + os.path.splitext(model_file)[0] + '_normal_matrix.txt', 'w')
            f.write(str(cm_normalized))
            f.close()
            print 'generated report for ' + model_file


def plot_confusion_matrix(cm, labels, filename, cmap=plot.cm.Blues):
    figure = plot.figure()
    plot.imshow(cm, interpolation='nearest', cmap=cmap)
    plot.title('Confusion Matrix')
    plot.colorbar()
    tick_marks = np.arange(len(labels))
    plot.xticks(tick_marks, labels, rotation=45)
    plot.yticks(tick_marks, labels)
    plot.ylabel('True label')
    plot.xlabel('Predicted label')
    plot.tight_layout()
    plot.savefig(filename + '_confusion_matrix.png')


def plot_learning_rate(hist, filename):
    figure = plot.figure()
    plot.title("Categorical Crossentropy Loss")
    plot.plot(hist.history["loss"], "b-", label="training loss")
    plot.plot(hist.history["val_loss"], "g-", label="validation loss")
    plot.legend()
    plot.xlabel("Epochs")
    plot.tight_layout()
    plot.savefig(filename + '_learning_rate.png')


def save_model(model, path='./models', name='model'):
    json_string = model.to_json()
    open(path + name + '.json', 'w').write(json_string)
    model.save_weights(path + '/' + name + '_weights.h5')


def export_model(model, path='./models', name='model'):
    json_string = model.to_json()
    open(path + '/' + name + '.json', 'w').write(json_string)


def load_model(path):
    l_model = model_from_json(open(path + '.json').read())
    l_model.load_weights(path + '_weights.h5')
    return l_model


def import_model(path):
    l_model = model_from_json(open(path + '.json').read())
    return l_model

def predict_image(path, model):
    image = ''
    prediction = model.predict_classes(image, verbose=1)


#def parse_input():


if __name__ == "__main__":
    valid_split = 0.15
    augment_data = True
    greyscale = True
    check_path = './temp.h5'
    model_path = './models'
    model_name = "48-48-96-96-15pct-no-strides"
    train_data, train_labels, val_data, val_labels, test_data, test_labels, num_classes = load_data()
    lookup_labels = ['airplane', 'automobile', 'bird', 'cat',
                     'deer', 'dog', 'frog',
                     'horse', 'ship', 'truck']
    #generate_reports('./models', test_data, test_labels)
    model = create_cnn(train_data.shape[1], train_data.shape[2], train_data.shape[3], num_classes)
    model, history = train_model(model, train_data, train_labels, val_data, val_labels)
    model.load_weights(check_path)
    evaluate_model(model, history, test_data, test_labels, model_path, model_name)
    save_model(model, model_path, model_name)
