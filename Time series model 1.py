#!/usr/bin/env python
# coding: utf-8

# In[77]:


pwd


# In[78]:


#changing pwd
import os
os.chdir('C:\\Users\\Sameera\\Desktop\\Time_series_model\\ARIMA-And-Seasonal-ARIMA-master')


# In[79]:


pwd


# ARIMA and Seasonal ARIMA
# Autoregressive Integrated Moving Averages
# 
# The general process for ARIMA models is the following:
# Visualize the Time Series Data
# Make the time series data stationary
# Plot the Correlation and AutoCorrelation Charts
# Construct the ARIMA Model or Seasonal ARIMA based on the data
# Use the model to make predictions
# Let's go through these steps!

# In[80]:


import pandas as pd 
import numpy as np


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

df=pd.read_csv("perrin-freres-monthly-champagne.csv")


# In[81]:


df.info()


# In[82]:


df.head()


# In[83]:


df.tail()


# In[84]:


## Cleaning up the data
df.columns=["Month","Sales"]
df.head()


# In[85]:


## Drop last 2 rows
df.drop(106,axis=0,inplace=True)
df.drop(105,axis=0,inplace=True)


# In[86]:


df.tail()


# In[87]:


# Convert Month into Datetime
df['Month']=pd.to_datetime(df['Month'])


# In[88]:


df.head()


# In[89]:


df.set_index('Month',inplace=True)


# In[90]:


df.head()


# Step 2: Visualize the Data

# In[91]:


df.plot()


# In[92]:


### Testing For Stationarity

from statsmodels.tsa.stattools import adfuller


# In[93]:


#Ho: It is non stationary
#H1: It is stationary

def adfuller_test(sales):
    result=adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
    


# In[94]:


adfuller_test(df['Sales'])


# Differencing

# In[95]:


df['Sales First Difference'] = df['Sales'] - df['Sales'].shift(1)


# In[96]:


df['Sales'].shift(1)


# In[97]:


df['Seasonal First Difference']=df['Sales']-df['Sales'].shift(12)


# In[98]:


df.head(15)


# In[99]:


## Again test dickey fuller test
adfuller_test(df['Seasonal First Difference'].dropna())


# Auto Regressive Model

# In[100]:


df['Seasonal First Difference'].plot()


# In[101]:


from pandas.plotting import autocorrelation_plot
#import pandas.plotting.autocorrelation_plot
autocorrelation_plot(df['Sales'])
plt.show()


# Final Thoughts on Autocorrelation and Partial Autocorrelation
# Identification of an AR model is often best done with the PACF.
# For an AR model, the theoretical PACF “shuts off” past the order of the model. The phrase “shuts off” means that in theory the partial autocorrelations are equal to 0 beyond that point. Put another way, the number of non-zero partial autocorrelations gives the order of the AR model. By the “order of the model” we mean the most extreme lag of x that is used as a predictor.
# Identification of an MA model is often best done with the ACF rather than the PACF.
# 
# For an MA model, the theoretical PACF does not shut off, but instead tapers toward 0 in some manner. A clearer pattern for an MA model is in the ACF. The ACF will have non-zero autocorrelations only at lags involved in the model.
# 
# p,d,q p AR model lags d differencing q MA lags

# In[69]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api
import statsmodels as sm
import statsmodels.api as sm


# In[102]:


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['Seasonal First Difference'].iloc[13:],lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Seasonal First Difference'].iloc[13:],lags=40,ax=ax2)


# In[73]:


df.head()


# In[74]:


df.columns


# In[103]:


# For non-seasonal data
#p=1, d=1, q=0 or 1
from statsmodels.tsa.arima_model import ARIMA


# In[104]:


model=ARIMA(df['Sales'],order=(1,1,1))
model_fit=model.fit()


# In[105]:


model_fit.summary()


# In[106]:


df['forecast']=model_fit.predict(start=90,end=103,dynamic=True)
df[['Sales','forecast']].plot(figsize=(12,8))


# In[107]:


model=sm.tsa.statespace.SARIMAX(df['Sales'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
results=model.fit()


# In[108]:


df['forecast']=results.predict(start=90,end=103,dynamic=True)
df[['Sales','forecast']].plot(figsize=(12,8))


# In[109]:


from pandas.tseries.offsets import DateOffset
future_dates=[df.index[-1]+ DateOffset(months=x)for x in range(0,24)]


# In[110]:


future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df.columns)


# In[111]:


future_datest_df.tail()


# In[112]:


future_df=pd.concat([df,future_datest_df])


# In[113]:


future_df['forecast'] = results.predict(start = 104, end = 120, dynamic= True)  
future_df[['Sales', 'forecast']].plot(figsize=(12, 8)) 


# In[ ]:




