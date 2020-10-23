# Carlos Morato, PhD.
# cwmorato@wpi.edu
# Deep Learning for Advanced Robot Perception
#
# Load and Plot the IMDB dataset

import numpy
from keras.datasets import imdb
from matplotlib import pyplot
# load the dataset
top_words = 5000
# save np.load
np_load_old = numpy.load
# modify the default parameters of np.load
numpy.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
# call load_data with allow_pickle implicitly set to true
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)
# restore np.load for future normal usage
numpy.load = np_load_old# summarize size

print("Training data: ")
print(X_train.shape)
print(y_train.shape)
# Summarize number of classes
print("Classes: ")
print(numpy.unique(y_train))
# Summarize number of words
print("Number of words: ")
print(len(numpy.unique(numpy.hstack(X_train))))
# Summarize review length
print("Review length: ")
result = list(map(len, X_train))
print("Mean %.2f words (%f)" % (numpy.mean(result), numpy.std(result)))
# plot review length as a boxplot and histogram
pyplot.subplot(121)
pyplot.boxplot(result)
pyplot.subplot(122)
pyplot.hist(result)
pyplot.show()
