import re

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras



# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # containers for input/output pairs
    X = []
    y = []
    
    # get last item in list
    last_place = len(series)-window_size-1 # -1 as second last item will only be on y
    
    for i in range(last_place):
        x = series[i:(i+window_size)] # get all item in window_size
        X.append(x)
        y.append(series[i + window_size]) # get last item after window_size
    
        
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
    model = Sequential()
    
    #add LSTM with 7 hidden nodes
    model.add(LSTM(7, input_shape=(window_size,1)))
    
    #add fully connected module with one unit
    model.add(Dense(1))
    return model


### TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    # find all unique characters in the text
    #we removed all non-english charaters using regex
    text = re.sub(r'[^a-zA-Z\,\.\1\;\:\?\s\']' , '', text)

    # \Xa0 is non-breaking space in Latin1 (ISO 8859-1), also chr(160). So replace it with a space.
    text = text.replace('\xa0', '')

    # shorten any extra dead space created above
    text = text.replace('  ',' ')
	
    return text


### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    #get last place
    last_place = len(text) - window_size
    
    for i in range(0, last_place, step_size):
        x = i + window_size
        X = text[i : x]
        y = text[x] 

        inputs.append(X)
        outputs.append(y)

    return inputs,outputs

