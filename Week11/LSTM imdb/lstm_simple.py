# Carlos Morato, PhD.
# cwmorato@wpi.edu
# Deep Learning for Advanced Robot Perception
#
# LSTM for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Convolution1D, MaxPooling1D, Dropout
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.regularizers import l2
from theano.tensor.shared_randomstreams import RandomStreams
# fix random seed for reproducibility
numpy.random.seed(7)
srng = RandomStreams(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 500
test_split = 0.33
np_load_old = numpy.load
numpy.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
numpy.load = np_load_old
# truncate and pad input sequences
max_review_length = 400
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)


# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length, dropout=0.2))
# model.add(Convolution1D(filters=256, kernel_size=5, padding='same', activation='relu', kernel_regularizer=l2(l=0.01)))
model.add(Convolution1D(filters=256, kernel_size=3, padding='same', activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(3))
model.add(Dropout(0.5))
model.add(Convolution1D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(3))
model.add(Dropout(0.5))
model.add(Convolution1D(filters=64, kernel_size=5, padding='same', activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(3))
model.add(Dropout(0.5))
model.add(LSTM(64, dropout=0.5, recurrent_dropout=0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)