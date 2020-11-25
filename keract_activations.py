import numpy as np
from keract import get_activations
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, concatenate
import keract

# model definition
i1 = Input(shape=(10,), name='i1')
i2 = Input(shape=(10,), name='i2')

a = Dense(1, name='fc1')(i1)
b = Dense(1, name='fc2')(i2)

c = concatenate([a, b], name='concat')
d = Dense(1, name='out')(c)
model = Model(inputs=[i1, i2], outputs=[d])

# inputs to the model
x = [np.random.uniform(size=(32, 10)), np.random.uniform(size=(32, 10))]

# call to fetch the activations of the model.
activations = get_activations(model, x, auto_compile=True)

# print the activations shapes.
[print(k, '->', v.shape, '- Numpy array') for (k, v) in activations.items()]

# Print output:
# i1 -> (32, 10) - Numpy array
# i2 -> (32, 10) - Numpy array
# fc1 -> (32, 1) - Numpy array
# fc2 -> (32, 1) - Numpy array
# concat -> (32, 2) - Numpy array
# out -> (32, 1) - Numpy array

keract.display_activations(activations, cmap=None, save=False, directory='.', data_format='channels_last', fig_size=(24, 24), reshape_1d_layers=False)
