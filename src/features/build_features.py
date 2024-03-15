import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('/home/kssocha/Desktop/Nauka/portfolio/2024-alk-ml-lstm-model/data/raw/yf_data.csv', index_col='Date')

#plt.figure(1, figsize=(16,4))
#plt.plot(df['gold_price'])

#calculate the percentage change
df['returns'] = df.pct_change()

#plt.figure(1, figsize=(16,4))
#plt.plot(df['returns'])

#calculate the log returns
#https://quantivity.wordpress.com/2011/02/21/why-log-returns/
df['log_returns'] = np.log(1 + df['returns'])

#plt.figure(1, figsize=(16,4))
#plt.plot(df['log_returns'])

#drop rows with missing values
df = df.dropna(how='any')

#create feature matrix X
X = df[['gold_price', 'log_returns']].values

#data normalization with MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
X_scaled = scaler.transform(X)

#create target vector y
y = [x[0] for x in X_scaled]

#data split into train and test
split_1 = int(len(X_scaled) * 0.7)
split_2 = int(len(X_scaled) * 0.9)

X_train = X_scaled[:split_1]
X_validation = X_scaled[split_1 : split_2]
X_test = X_scaled[split_2 : len(X_scaled)]
y_train = y[:split_1]
y_validation = y[split_1 : split_2]
y_test = y[split_2 : len(y)]

#test the lengths
assert len(X_train) == len(y_train)
assert len(X_validation) == len(y_validation)
assert len(X_test) == len(y_test)

###labeling

n = 3
Xtrain = []
ytrain = []
Xvalidation = []
yvalidation = []
Xtest = []
ytest = []
for i in range(n, len(X_train)):
    Xtrain.append(X_train[i - n: i, : X_train.shape[1]])
    ytrain.append(y_train[i])
for i in range(n,len(X_validation)):
    Xvalidation.append(X_validation[i - n: i, : X_validation.shape[1]])
    yvalidation.append(y_validation[i])
for i in range(n, len(X_test)):
    Xtest.append(X_test[i - n: i, : X_test.shape[1]])
    ytest.append(y_test[i])
    
#revers transformation
val = np.array(ytrain[0])
val = np.c_[val, np.zeros(val.shape)]
result = scaler.inverse_transform(val)

#LSTM inputs reshaping
Xtrain, ytrain = (np.array(Xtrain), np.array(ytrain))
Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], Xtrain.shape[2]))

Xvalidation, yvalidation = (np.array(Xvalidation), np.array(yvalidation))
Xvalidation = np.reshape(Xvalidation, (Xvalidation.shape[0], Xvalidation.shape[1], Xvalidation.shape[2]))

Xtest, ytest = (np.array(Xtest), np.array(ytest))
Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], Xtest.shape[2]))

# print(Xtrain.shape)
# print(ytrain.shape)
# print(Xvalidation.shape)
# print(yvalidation.shape)
# print(Xtest.shape)
# print(ytest.shape)