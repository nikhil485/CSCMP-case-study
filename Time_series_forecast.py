#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 18:47:25 2020

@author: nikdesh
"""

import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'
from pylab import rcParams
from sklearn.metrics import mean_squared_error
from math import sqrt


df = pd.read_excel("/Users/nikdesh/Documents/Condata/CSCMP/Historical Product Demand.xlsx")

# Drop columns with zeroes
df = df.dropna()

# Choose only the rows with product_1248
product_1248 = df.loc[df['Product_Code']=='Product_1248']

# Drop the columns you don't need. Since only one product is selected,
# Product_Code column is not needed.

product_1248 = product_1248[['Date', 'Order_Demand']]

# Sort by date
product_1248 = product_1248.sort_values('Date')

# This will add the order demands that fall on the same day
product_1248 = product_1248.groupby('Date')['Order_Demand'].sum().reset_index()

# This converts the date column to datetime, useful for searching through dates later
product_1248['Date'] = pd.to_datetime(product_1248['Date'])
product_1248 = product_1248.set_index('Date')
product_1248.index

# This prevents the training model from seeing the 2016 data.
# This is important since the 2016 data is going to be used as a validation set
product_1248_train = product_1248.loc['2012-01-01':'2015-12-31']
product_1248_validation = product_1248.loc['2012-01-01':"2016-12-31"]

# This groups by month, and then sums the group
y_train = product_1248_train.groupby(pd.Grouper(freq="M")).sum()
y_train['2012':]

y_validation = product_1248_validation.groupby(pd.Grouper(freq="M")).sum()

# A general time-series plot
y_train.plot(figsize=(15,6))
plt.show()

# Seasonality analysis

rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y_train, model='additive')
fig = decomposition.plot()
plt.show()

# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

pdq = list(itertools.product(p, d, q))

seasonal_pdq = [(x[0], x[1], x[2], 6) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

warnings.filterwarnings("ignore") # specify to ignore warning messages

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y_train,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
        
mod = sm.tsa.statespace.SARIMAX(y_train,
                                order=(1, 1, 0),
                                seasonal_order=(1, 1, 1, 6),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit(disp=0)

# print(results.summary().tables[1])

results.plot_diagnostics(figsize=(15, 12))
plt.show()

# ---------------
# pred_dynamic = results.get_prediction(start=pd.to_datetime('2016-01-31'), dynamic=True, full_results=True)
# pred_dynamic_ci = pred_dynamic.conf_int()

# ax = y_validation['2016':].plot(label='observed', figsize=(20, 15))
# pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)

# ax.fill_between(pred_dynamic_ci.index,
                # pred_dynamic_ci.iloc[:, 0],
                # pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)

# ax.fill_betweenx(ax.get_ylim(), pd.to_datetime('2016-01-31'), y_validation.index[-1],
                 # alpha=.1, zorder=-1)

# ax.set_xlabel('Date')
# ax.set_ylabel('Order_Demand')

# plt.legend()
# plt.show()




pred_uc = results.get_forecast(steps=12)

# Get confidence intervals of forecasts
pred_ci = pred_uc.conf_int()

ax = y_validation.plot(label='observed', figsize=(20, 15))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Order Demand')

plt.legend()
plt.show()

#y_forecasted = pred_uc.predicted_mean
#y_truth = y_validation['2016-01-31':]

# Compute the mean square error
#mse = mean_squared_error(y_truth, y_forecasted)
#print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
