import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    X = []
    y = []
    l = 0
    while (l +window_size) < len(series):
        X.append(series[l:window_size+l])
        y.append(series[l+window_size])
        l += 1

    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X, y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']

    import string
    alphabet = string.ascii_lowercase + ' '
    punctuation = ['!', ',', '.', ':', ';', '?']
    keepchars = alphabet + ''.join(punctuation)
    #convert to list and replace unwanted chars with blanks
    listtext = list(text)
    for i,char in enumerate(listtext):
        if char not in keepchars:
            listtext[i] = ' '
    text = ''.join(listtext)
    return text


### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    l = 0
    while (l +window_size) < len(text):
        inputs.append(text[l:window_size+l])
        outputs.append(text[l+window_size])
        l += step_size

    return inputs,outputs

# TODO build the required RNN model:
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss
def build_part2_RNN(window_size, num_chars):

    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars, activation ="softmax" ))
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model
