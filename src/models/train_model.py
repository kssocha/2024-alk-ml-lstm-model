import os
sys_path = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import sys
sys.path.append(sys_path)

#import features
from src.features.build_features import Xtrain, ytrain, Xvalidation, yvalidation

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import save_model

from sklearn.metrics import root_mean_squared_error

#LSTM model
model = Sequential()
#1st LSTM layer with 5 neurons, dropout 20%
#dropout - regularization technique to prevent overfitting
#by randomly dropping a fraction of neurons during training (20% of neurons will be set to zero)
model.add(LSTM(5, input_shape=(Xtrain.shape[1], Xtrain.shape[2]), dropout=0.2, return_sequences=True))
#2nd LSTM layer with 10 neurons, dropout 20%
model.add(LSTM(10, input_shape=(Xtrain.shape[1], Xtrain.shape[2]), dropout=0.2))
#1st dense layer with 5 neurons
#relu - Rectified Linear Activation, it replaces all negative values in the input tensor with zero
#and maintains all positive values
model.add(Dense(5, activation='relu'))
#output dense layer with 1 neuron
model.add(Dense(1))
#loss function specifies the function to minimize
#optimizer specifies the algorithm to use to minimize the loss function
#adam - Adaptive Moment Estimation, it is a method for stochastic optimization
#which dinamically updates the learning rate
model.compile(loss='mean_squared_error', optimizer='adam')
#verbose - amount of info printed during the training, verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch
model.fit(
    Xtrain, ytrain, epochs=20000, validation_data=(Xvalidation, yvalidation), batch_size=32, verbose=1
)

#save model
save_model(model, 'lstm_model.h5')