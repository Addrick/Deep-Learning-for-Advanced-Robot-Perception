# Carlos Morato, PhD.
# cwmorato@wpi.edu
# Deep Learning for Advanced Robot Perception
#
# Visualize training history
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.constraints import maxnorm
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataframe = pandas.read_csv("sonar.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# create model
model = Sequential()
model.add(Dropout(0.2, input_shape=(60,)))
model.add(Dense(60, activation="sigmoid", kernel_initializer="normal", kernel_constraint=maxnorm(3)))
model.add(Dropout(0.3))
model.add(Dense(60, activation="sigmoid", kernel_initializer="normal", kernel_constraint=maxnorm(3)))
model.add(Dropout(0.4))
model.add(Dense(60, activation="sigmoid", kernel_initializer="normal", kernel_constraint=maxnorm(3)))
model.add(Dropout(0.4))
# model.add(Dense(30, activation="sigmoid", kernel_initializer="normal", kernel_constraint=maxnorm(3)))
# model.add(Dropout(0.3))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# Compile model
sgd = SGD(lr=0.15, momentum=0.7, decay=0.00005, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
# Fit the model
history = model.fit(X, Y, validation_split=0.33, nb_epoch=150, batch_size=10, verbose=0)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
