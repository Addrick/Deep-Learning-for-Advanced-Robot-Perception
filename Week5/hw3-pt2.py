# Adam Santos, BS
# asantos@wpi.edu
#
# Adapted code originally from:
# Carlos Morato, PhD.
# cwmorato@wpi.edu
# Deep Learning for Advanced Robot Perception
#
# Current best configuration: Test accuracy: 88.45% (6.48%)
# (0.2)>Dense(60, sig)>(0.3)>Dense(60, sig)>(0.4)>Dense(60,sig)>(0.4)>Dense(1,sig), 
# lr=0.15, momentum=0.7, decay=0.00005, 2000 epochs
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.utils import normalize
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

from keras.callbacks import History
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

# dropout in the input layer with weight constraint
def create_model():
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
    return model

history=[History()]
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_model, epochs=2000, batch_size=16, verbose=2)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(y=encoded_Y, n_folds=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold, fit_params={'mlp__callbacks':history})
print("Test accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
