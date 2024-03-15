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

import matplotlib.pyplot as plt

#import data
from src.features.build_features import df, n, split_1, split_2, X_scaled

#data split into train and test
df_train = df[n:split_1]
df_validation = df[split_1+n : split_2]
df_test = df[split_2+n : len(X_scaled)]

try:
    test_predict = np.load('/home/kssocha/Desktop/Nauka/portfolio/2024-alk-ml-lstm-model/src/models/test_predict.npy')
except FileNotFoundError:
    print('File not found')    

df_test['predictions'] = test_predict

#plot actual vs predicted values
plt.plot(df_train.iloc[-(int(0.25*len(df_train))):]['gold_price'], label='25% of Train')
plt.plot(df_validation['gold_price'], label='Validation')
plt.plot(df_test['gold_price'], label='Actual')
plt.plot(df_test['predictions'], label='Forecast')
plt.legend(loc='upper left')
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(20))
plt.xticks(rotation=90)
plt.show();