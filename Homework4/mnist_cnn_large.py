# Carlos Morato, PhD.
# cwmorato@wpi.edu
# Deep Learning for Advanced Robot Perception
#
# Larger CNN for the MNIST Dataset
import numpy
from keras.constraints import maxnorm
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import tensorflow as tf
import plot_history as p
# allocate ~4GB of GPU memory:
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.67)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][width][height][channels]
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
# define the larger model
def larger_model():
	# create model
	model = Sequential()
	model.add(Dropout(0.1, input_shape=(28, 28, 1)))
	model.add(Convolution2D(56, 5, 5, border_mode='valid', activation='relu', W_constraint=maxnorm(3)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(28, 3, 3, activation='relu', W_constraint=maxnorm(3)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	# model.add(Convolution2D(14, 3, 3, activation='relu', W_constraint=maxnorm(3)))
	# model.add(MaxPooling2D(pool_size=(2, 2)))
	# model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
# build the model
model = larger_model()
# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=75, batch_size=128, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))
p.plot_acc_loss(history)