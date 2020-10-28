# Carlos Morato, PhD.
# cwmorato@wpi.edu
# Deep Learning for Advanced Robot Perception
#
# ZCA whitening
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy
from keras.constraints import maxnorm
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Dropout
from keras.optimizers import SGD, Adagrad
from keras.utils import np_utils
import matplotlib.pyplot as plt
num_pixels = 32*32
num_classes = 10
def baseline_model():
    # create model
    model = Sequential()
    # model.add(Flatten(input_shape=(28, 28, 1)))
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(784, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(4)))
    model.add(Dropout(0.5))
    model.add(Dense(784, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(98, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='relu'))
    # Compile model
    # sgd = SGD(lr=0.1, momentum=0.7, decay=0, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
# convert from int to float
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# define data preparation
datagen = ImageDataGenerator(zca_whitening=True, zca_epsilon=1e-54)
# fit parameters from data
datagen.fit(X_train)

model = baseline_model()
for e in range(32):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in datagen.flow(X_train, y_train, batch_size=1):
        model.fit(x_batch, y_batch)
        batches += 1
        if batches >= len(X_train) / 32:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
