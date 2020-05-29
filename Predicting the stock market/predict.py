import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from datetime import datetime

# Load the historical stock market from 1950 to 2015
stocks = pd.read_csv('sphist.csv')

# Convert dates to datetime objects and sort in ascending order
stocks['Date'] = pd.to_datetime(stocks['Date'].copy())
stocks = stocks.sort_values(by='Date')


# We will use 'Close', the closing price for the day once trading is finished, as the stock price.
# Here are the indicators that will help us make predictions:

# 1 - The average price from the past 5 days.
stocks['avg5'] = pd.Series()
# 2 - The average price for the past 365 days.
stocks['avg365'] = pd.Series()
# 3 - The standard deviation of the price over the past 5 days.
stocks['std5'] = pd.Series()
# 4 - The standard deviation of the price over the past 365 days.
stocks['std365'] = pd.Series()
# 5 - The average volume for the past 5 days.
stocks['vol5'] = pd.Series()
# 6 - The average volume for the past 365 days.
stocks['vol365'] = pd.Series()

# Rolling average
for ii in range(len(stocks.index)):
    try:
        stocks.at[stocks.index[ii], 'avg5'] = stocks.loc[stocks.index[ii-5:ii], 'Close'].mean()
        stocks.at[stocks.index[ii], 'avg365'] = stocks.loc[stocks.index[ii-365:ii], 'Close'].mean()
        stocks.at[stocks.index[ii], 'std5'] = stocks.loc[stocks.index[ii-5:ii], 'Close'].std()
        stocks.at[stocks.index[ii], 'std365'] = stocks.loc[stocks.index[ii-365:ii], 'Close'].std()
        stocks.at[stocks.index[ii], 'vol5'] = stocks.loc[stocks.index[ii-5:ii], 'Volume'].mean()
        stocks.at[stocks.index[ii], 'vol365'] = stocks.loc[stocks.index[ii-365:ii], 'Volume'].mean()
    except IndexError:
        pass

# Drop the first year (can calculate avg365) (there are no additional null values but we would drop them anyway)
stocks.dropna(axis=0, inplace=True)

train = stocks[stocks['Date'] < datetime(2013, 1, 1)]
test  = stocks[stocks['Date'] >= datetime(2013, 1, 1)]


def train_and_test_linear_model(features):
    '''Takes feature columns as an input and trains a linear regression model on target 'Close' (closing stock price).'''
    lr = LinearRegression()
    lr.fit(train[features], train['Close'])
    predictions = lr.predict(test[features])

    # Calculate mean absolute error (mae)
    mae = mean_absolute_error(test['Close'], predictions)
    return mae

# Let's start by using only the average columns
features = [
    'avg5', 
    'avg365', 
]
mae1 = train_and_test_linear_model(features)

# Then we can add the std columns and see if that affects the mae
features += [
    'std5', 
    'std365', 
]
mae2 = train_and_test_linear_model(features)

# Finally, let's add the volume columns and see if that affects the mae
features += [
    'vol5', 
    'vol365', 
]
mae3 = train_and_test_linear_model(features)

print(mae1, mae2, mae3)

# Result : 16.130 16.130 16.143
# We don't gain accuracy when adding close price standard deviation or stock volume. 
# If we use solely those features instead of appending them, the results are:
# 16.13 758.72 700.39
# which shows how poorly std and volume performs compared to avg.