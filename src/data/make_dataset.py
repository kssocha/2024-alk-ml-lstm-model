from datetime import date
from dateutil.relativedelta import relativedelta

import pandas as pd

import yfinance as yf

#time range for all queries
start_date = (date.today() - relativedelta(years = 25)).strftime('%Y-%m-%d')
end_date = date.today().strftime('%Y-%m-%d')

#get the gold and silver price USD/oz from Yahoo Finance API
df = yf.download(['GC=F'],
                    start = start_date,
                    end = end_date,
                    progress = False)

#drop the columns that are not needed
df = df.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis = 1)

#drop the rows with missing values
df = df.dropna(how = 'any')

#rename the columns
df.columns = ['gold_price']

#write the data to a csv file
df.to_csv('/home/kssocha/Desktop/Nauka/portfolio/2024-alk-ml-lstm-model/data/raw/yf_data.csv')