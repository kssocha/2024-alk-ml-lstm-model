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
from src.features.build_features import Xtrain, Xvalidation, Xtest, scaler

import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import root_mean_squared_error

#load model
model = load_model('lstm_model.h5')

#prediction
train_predict = model.predict(Xtrain)
validation_predict = model.predict(Xvalidation)
test_predict = model.predict(Xtest)

train_predict = np.c_[train_predict, np.zeros(train_predict.shape)]
validation_predict = np.c_[validation_predict, np.zeros(validation_predict.shape)]
test_predict = np.c_[test_predict, np.zeros(test_predict.shape)]

#invert prediction
train_predict = scaler.inverse_transform(train_predict)
train_predict = [x[0] for x in train_predict]

validation_predict = scaler.inverse_transform(validation_predict)
validation_predict = [x[0] for x in validation_predict]

test_predict = scaler.inverse_transform(test_predict)
test_predict = [x[0] for x in test_predict]

#calculate square root of mean squared error
train_score = root_mean_squared_error([x[0][0] for x in Xtrain], train_predict)
print('Train Score: %.2f RMSE' % (train_score))

validation_score = root_mean_squared_error([x[0][0] for x in Xvalidation], validation_predict)
print('Validation Score: %.2f RMSE' % (validation_score))

test_score = root_mean_squared_error([x[0][0] for x in Xtest], test_predict)
print('Test Score: %.2f RMSE' % (test_score))

np.save(('test_predict.npy'), test_predict)