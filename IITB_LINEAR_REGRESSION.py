# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 11:38:14 2021

@author: Abhay
"""
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

'''
:::::: Importing Stocks data from Yahoo :::::::::
'''
ticker = ['^NSEBANK','AUBANK.NS','AXISBANK.NS','BANDHANBNK.NS','FEDERALBNK.NS','HDFCBANK.NS','ICICIBANK.NS','IDFCFIRSTB.NS','INDUSINDBK.NS','KOTAKBANK.NS','PNB.NS','RBLBANK.NS','SBIN.NS']
raw_data = pd.DataFrame(columns=ticker)
for i in range(len(ticker)):
    download = []
    download = yf.download(ticker[i], period='2y',interval='1d')
    raw_data[ticker[i]] = download['Adj Close'] 
idx = download.index
raw_data.reset_index(inplace=True,drop=False)
raw_data.dropna(axis=0,how="any")
raw_data.head()

'''
:::::: Converting Raw Prices into return series to do analysis on stationary data. ::::::

'''
data_ret = pd.DataFrame(columns=ticker)
for i in range(len(ticker)):
    data_ret[ticker[i]] = raw_data[ticker[i]].pct_change()
data_ret.index = raw_data['Date']
data_ret.reset_index(inplace=True,drop=False)
data_ret.dropna(axis=0,how="any",inplace=True)
data_ret.head()
data_ret.columns

'''
::::: Plotting Raw Returns Data ::::::

'''

fig, ((ax11,ax12),(ax21,ax22),(ax31,ax32),(ax41,ax42),(ax51,ax52),(ax61,ax62)) = plt.subplots(6,2,figsize = (24,12))
fig.constrained_layout = True

ax11.scatter(data_ret['AUBANK.NS'],data_ret['^NSEBANK'])
ax11.set_title('AU BANK v/s Bank Nifty')

ax12.scatter(data_ret['AXISBANK.NS'],data_ret['^NSEBANK'])
ax12.set_title('AXIS BANK v/s Bank Nifty')

ax21.scatter(data_ret['BANDHANBNK.NS'],data_ret['^NSEBANK'])
ax21.set_title('BANDHAN BANK v/s Bank Nifty')

ax22.scatter(data_ret['FEDERALBNK.NS'],data_ret['^NSEBANK'])
ax22.set_title('FEDERAL BANK v/s Bank Nifty')

ax31.scatter(data_ret['HDFCBANK.NS'],data_ret['^NSEBANK'])
ax31.set_title('HDFC BANK v/s Bank Nifty')

ax32.scatter(data_ret['ICICIBANK.NS'],data_ret['^NSEBANK'],)
ax32.set_title('ICICI BANK v/s Bank Nifty')

ax41.scatter(data_ret['IDFCFIRSTB.NS'],data_ret['^NSEBANK'])
ax41.set_title('IDFC FIRST BANK v/s Bank Nifty')

ax42.scatter(data_ret['INDUSINDBK.NS'],data_ret['^NSEBANK'])
ax42.set_title('INDUSIND BANK v/s Bank Nifty')

ax51.scatter(data_ret['KOTAKBANK.NS'],data_ret['^NSEBANK'])
ax51.set_title('KOTAK BANK v/s Bank Nifty')

ax52.scatter(data_ret['PNB.NS'],data_ret['^NSEBANK'])
ax52.set_title('PNB BANK v/s Bank Nifty')

ax61.scatter(data_ret['RBLBANK.NS'],data_ret['^NSEBANK'])
ax61.set_title('RBL BANK v/s Bank Nifty')

ax62.scatter(data_ret['SBIN.NS'],data_ret['^NSEBANK'])
ax62.set_title('SBI BANK v/s Bank Nifty')

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()

# Let's compute Beta Coeeficient
n = len(data_ret.index)
X_data = data_ret[data_ret.columns[2:]]
X_array = data_ret[data_ret.columns[2:]].to_numpy()
X = np.hstack((X_array, np.ones((X_array.shape[0],1),dtype=X_array.dtype)))

# Convert Predictor Variable Columns into Numpy Array
response_col = [1]
y = data_ret[data_ret.columns[response_col]].to_numpy()

# Calculate Beta for Regression
XTX = np.matmul(np.transpose(X),X)
Xy = np.matmul(np.transpose(X),y)
beta = np.matmul(np.linalg.inv(XTX),Xy)

# Create Residual plots
e_i_residuals_list = []
for i in range(n):
    x_i = X[i,:]
    y_i = y[i]
    y_pred_i = np.dot(x_i, beta)
    e_i = y_i - y_pred_i
    e_i_residuals_list.append(e_i)
plt.scatter(data_ret['^NSEBANK'],e_i_residuals_list,color='r')
plt.title('Residual Plot')
plt.xlabel('Bank Nifty')
plt.ylabel('Residuals')
plt.grid()
plt.show()

# Let's Compute R^2
y_bar = 0
for i in range(n):
    y_i = float(data_ret.iat[i,1])
    y_bar += y_i
y_bar /= n
print('Y_Bar :',y_bar)

sigma_yy = 0
for i in range(n):
    y_i = float(data_ret.iat[i,1])
    sigma_yy += (y_i-y_bar)**2
print('Sigma YY :', sigma_yy)

# Calculate sum of squared residuals
sum_sq_residuals = 0
for i in range(n):
    x_i = X[i,:]
    y_i = y[i]
    y_pred_i = np.dot(x_i,beta)
    e_i = y_i - y_pred_i 
    sum_sq_residuals += (e_i)**2
print('Sum of Squared Errors :', sum_sq_residuals)

# Calculating R^2
R_sq = 1 - sum_sq_residuals/sigma_yy
print('R^2 :',R_sq)





















