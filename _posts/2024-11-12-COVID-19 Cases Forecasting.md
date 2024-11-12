---
layout: single
title:  "COVID-19 Forecasting"
subtitle: "Time Series"
categories: python
tag: [Time Series, Rolling Mean, Rolling Standard Deviation, Log Return, Augmented Dicket-Fuller Test, Autocorrelations, ARMA on Log Returns, ARMA on First-order Differences, Vector Autoregression(VAR), Independent AR Models, Granger Causality]
toc: true
---


## 2020 Forecasting Daily New COVID-19 Cases

### Resources
- [Johns Hopkins University CSSE COVID-19](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series)

### Background
In this project, we are utilizing the same dataset that we used in **COVID-19 Cases Time Series** project. We aim to forecast daily new COVID-19 cases and test for causality among the time series of different countries. The analysis includes single and multiple times series forecasting, utilizing ARIMA and VAR models. The data analyzed comprises daily new COVID-19 cases worldwide and across the top 5 countries with the highest cumulative cases as of August 21, 2020

### Data
The dataset contains COVID data on 6 countries of interest:
- **Country/Region**: There are 6 countries including Brazil, India, Russia, South Africa, South Korea and United States
- **Date**: The date range is from 2020/1/5 ~ 2020/8/21
- **# of COVID cases reported**: The data contains the number of cumulative confirmed cases globally as of certain dates


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import warnings
from statsmodels.tools.sm_exceptions import ValueWarning
warnings.simplefilter("ignore", ValueWarning)
warnings.filterwarnings("ignore")

df = pd.read_csv("C:/Users/shincho/OneDrive - KLA Corporation/Desktop/Portfolio/Time_Series2/time_series_covid19_confirmed_global.csv")
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Province/State</th>
      <th>Country/Region</th>
      <th>Lat</th>
      <th>Long</th>
      <th>1/22/20</th>
      <th>1/23/20</th>
      <th>1/24/20</th>
      <th>1/25/20</th>
      <th>1/26/20</th>
      <th>1/27/20</th>
      <th>...</th>
      <th>8/12/20</th>
      <th>8/13/20</th>
      <th>8/14/20</th>
      <th>8/15/20</th>
      <th>8/16/20</th>
      <th>8/17/20</th>
      <th>8/18/20</th>
      <th>8/19/20</th>
      <th>8/20/20</th>
      <th>8/21/20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>Afghanistan</td>
      <td>33.939110</td>
      <td>67.709953</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>37345</td>
      <td>37424</td>
      <td>37431</td>
      <td>37551</td>
      <td>37596</td>
      <td>37599</td>
      <td>37599</td>
      <td>37599</td>
      <td>37856</td>
      <td>37894</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>Albania</td>
      <td>41.153300</td>
      <td>20.168300</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>6817</td>
      <td>6971</td>
      <td>7117</td>
      <td>7260</td>
      <td>7380</td>
      <td>7499</td>
      <td>7654</td>
      <td>7812</td>
      <td>7967</td>
      <td>8119</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>Algeria</td>
      <td>28.033900</td>
      <td>1.659600</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>36699</td>
      <td>37187</td>
      <td>37664</td>
      <td>38133</td>
      <td>38583</td>
      <td>39025</td>
      <td>39444</td>
      <td>39847</td>
      <td>40258</td>
      <td>40667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>Andorra</td>
      <td>42.506300</td>
      <td>1.521800</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>977</td>
      <td>981</td>
      <td>989</td>
      <td>989</td>
      <td>989</td>
      <td>1005</td>
      <td>1005</td>
      <td>1024</td>
      <td>1024</td>
      <td>1045</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>Angola</td>
      <td>-11.202700</td>
      <td>17.873900</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1762</td>
      <td>1815</td>
      <td>1852</td>
      <td>1879</td>
      <td>1906</td>
      <td>1935</td>
      <td>1966</td>
      <td>2015</td>
      <td>2044</td>
      <td>2068</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>261</th>
      <td>NaN</td>
      <td>West Bank and Gaza</td>
      <td>31.952200</td>
      <td>35.233200</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>15184</td>
      <td>15491</td>
      <td>15834</td>
      <td>16153</td>
      <td>16534</td>
      <td>16844</td>
      <td>17306</td>
      <td>17606</td>
      <td>17989</td>
      <td>18313</td>
    </tr>
    <tr>
      <th>262</th>
      <td>NaN</td>
      <td>Western Sahara</td>
      <td>24.215500</td>
      <td>-12.885800</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>263</th>
      <td>NaN</td>
      <td>Yemen</td>
      <td>15.552727</td>
      <td>48.516388</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1841</td>
      <td>1847</td>
      <td>1858</td>
      <td>1858</td>
      <td>1869</td>
      <td>1882</td>
      <td>1889</td>
      <td>1892</td>
      <td>1899</td>
      <td>1906</td>
    </tr>
    <tr>
      <th>264</th>
      <td>NaN</td>
      <td>Zambia</td>
      <td>-13.133897</td>
      <td>27.849332</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>8501</td>
      <td>8663</td>
      <td>9021</td>
      <td>9186</td>
      <td>9343</td>
      <td>9839</td>
      <td>9981</td>
      <td>10218</td>
      <td>10372</td>
      <td>10627</td>
    </tr>
    <tr>
      <th>265</th>
      <td>NaN</td>
      <td>Zimbabwe</td>
      <td>-19.015438</td>
      <td>29.154857</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>4893</td>
      <td>4990</td>
      <td>5072</td>
      <td>5176</td>
      <td>5261</td>
      <td>5308</td>
      <td>5378</td>
      <td>5643</td>
      <td>5745</td>
      <td>5815</td>
    </tr>
  </tbody>
</table>
<p>266 rows × 217 columns</p>
</div>




```python
def load_data():
    
    daily_new_cases = df.iloc[:, 4:].sum(axis=0)
    daily_new_cases.index = pd.to_datetime(daily_new_cases.index)
    daily_new_cases = daily_new_cases.diff().iloc[1:]
    
    return daily_new_cases

load_data()
```




    2020-01-23        99.0
    2020-01-24       287.0
    2020-01-25       493.0
    2020-01-26       684.0
    2020-01-27       809.0
                    ...   
    2020-08-17    209672.0
    2020-08-18    255096.0
    2020-08-19    274346.0
    2020-08-20    267183.0
    2020-08-21    270751.0
    Length: 212, dtype: float64



## Part 1: Stationarity Tests
Stationarity is a property of time series where statistical properties such as mean, variance, and autocorrelation are constant over time. Many statistical modeling methods assume stationarity as a predecessor for reliable forecasting

### 1a. Rolling Mean and Rolling Standard Deviation
We explore stationarity to understand whether our time series is stable over time. A stationary time series has constant statistical properties such as mean and variance. We compute the rolling mean and rolling standard deviation to check is these statistics are constant


```python
def calc_rolling_stats(ser, wd_size=7):
    
    weights = np.ones(wd_size)
    mean_values = [
        np.average(
            ser.iloc[max(0, x - wd_size + 1):x + 1], 
            weights=weights[-(x + 1 - max(0, x - wd_size + 1)):]
        ) for x in range(len(ser))
    ]
    mean = np.array(mean_values)
    squared_ser = ser**2
    mean_squared_values = [
        np.average(
            squared_ser.iloc[max(0, x - wd_size + 1):x + 1], 
            weights=weights[-(x + 1 - max(0, x - wd_size + 1)):]
        ) for x in range(len(squared_ser))
    ]
    mean_squared = np.array(mean_squared_values)
    std = np.sqrt(mean_squared - mean**2)

    return mean, std
```


```python
ser, wd_size = load_data(), 7
rolling_mean, rolling_std = calc_rolling_stats(ser, wd_size)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ser, label="Original")
ax.plot(pd.Series(rolling_mean, index=ser.index), label="Rolling Mean")
ax.plot(pd.Series(rolling_std, index=ser.index), label="Rolling Std")

ax.set_xlabel("Day")
ax.set_ylabel("# Cases")
ax.set_title("Daily New COVID-19 Cases Worldwide\n" + f"Rolling Stats with Window Size = {wd_size} Days")
ax.legend()

del fig, ax, ser, wd_size, rolling_mean, rolling_std
```


    
![output_5_0](https://github.com/user-attachments/assets/522f34bc-233e-45da-800b-d4813fdb7d7c)

    


### 1b. Log Return
We compute the log return of our time series. Log returns are used in time series analysis as they can help in stabilizing the variance, making the series more stationary. This transformation is particularly useful in epidemic data where values can vary over several orders of magnitude.

By comparing the rolling statistics of the log returns with that of the original series, we can see if the log transformation has made the series more stationary, thereby confirming its suitability for further time series modeling.


```python
def calc_log_ret(ser):
    
    log_ret = np.log(ser/ser.shift(1)).dropna()
    
    return log_ret
```


```python
log_ret, wd_size = calc_log_ret(load_data()), 7
rolling_mean, rolling_std = calc_rolling_stats(log_ret, wd_size)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(log_ret, label="Log Return")
ax.plot(pd.Series(rolling_mean, index=log_ret.index), label="Rolling Mean")
ax.plot(pd.Series(rolling_std, index=log_ret.index), label="Rolling Std")

ax.set_xlabel("Day")
ax.set_title("Log Return of Daily New COVID-19 Cases Worldwide\n" + f"Rolling Stats with Window Size = {wd_size} Days")
ax.legend()

del fig, ax, log_ret, wd_size, rolling_mean, rolling_std
```



![output_8_0](https://github.com/user-attachments/assets/f29ad6d3-9c77-42c7-a436-5a25846be0dc)


    


## Part 2: Augmented Dickey-Fuller Test
We perform the Augmented Dickey-Fuller (ADF) test to statistically determine stationarity. The null hypothesis of the ADF test is that the series is non-stationary. A p-value less than 0.05 indicates that we can reject the null hypothesis.


```python
from statsmodels.tsa.stattools import adfuller
_, pval, *_ = adfuller(load_data())
print(f"p-value: {pval}")

del adfuller, pval
```

    p-value: 0.6765852511544105
    

**Interpretation**
- Since the p-value is significantly greater than 0.05, we fail to reject the null hypothesis. This means that our initial time series is non-stationary, supporting our earlier visual analysis

## Part 3: Autocorrelations
Autocorrelation measures the correlation of a time series with its past values. The partial autocorrelation function (PACF) removes the influence of correlations at shorter lags to identify the extent of direct correlations


```python
#ACF Calculation
from statsmodels.tsa.stattools import acf

def calc_acf(ser,max_lag):
    
    acf_values = acf(ser, nlags=max_lag)
    
    return acf_values

#PACF Calculation
from statsmodels.tsa.stattools import pacf

def calc_pacf(ser, max_lag):
    
    ans_pacf = pacf(ser, nlags=max_lag)
    
    return ans_pacf
```


```python
from statsmodels.graphics.tsaplots import plot_acf

ser, max_lag = load_data(), 30

fig, ax = plt.subplots(1,1,figsize=(8,6))

plot_acf(ser,ax,lags=max_lag,title = "Daily New COFIC-19 Cases for Worldwide\nAutocorrelation Function")
ax.set_xlabel(r"Lag")
ax.set_ylabel(r"Correlation")

del fig, ax, ser, max_lag, plot_acf
```


    
![output_13_0](https://github.com/user-attachments/assets/7486c4be-802e-4f1e-8f88-9b9d72a49585)

    



```python
from statsmodels.graphics.tsaplots import plot_pacf

ser, max_lag = load_data(), 30

fig, ax = plt.subplots(1,1,figsize=(8,6))

plot_pacf(ser,ax,lags=max_lag,title="Daily New COVID-19 Cases Worldwide\nPartial Autocorrelation Function")
ax.set_xlabel(r"Lag")
ax.set_ylabel(r"Correlation")

del fig, ax, ser, max_lag, plot_pacf
```


    
![output_14_0](https://github.com/user-attachments/assets/862a5a9e-b962-4b43-b3ec-1a1edeb211e6)

    


## Part 4: ARMA on Log Returns
ARMA (AutoRegressive Moving Average) models combine autoregression and moving averages to capture serial dependencies in a time series. By using log returns, we aim to achieve stationarity, which is critical for reliable model performance. We fit an ARMA model to the log returns and use this model to make forecasts


```python
from statsmodels.tsa.arima.model import ARIMA

def arma_log_ret(ser, p, q, num_forecasts):
    
    ser_cleaned = ser.dropna()
    log_ret = np.log(ser_cleaned / ser_cleaned.shift(1)).dropna()
    arima_model = ARIMA(log_ret, order=(p, 0, q)).fit()
    log_ret_forecasts = arima_model.forecast(steps=num_forecasts)
    last_value = ser_cleaned.iloc[-1]
    forecast_values = [last_value]
    for log_ret in log_ret_forecasts:
        next_value = forecast_values[-1] * np.exp(log_ret)
        forecast_values.append(next_value)
    forecast_values = forecast_values[1:]
    last_date = ser_cleaned.index[-1]
    forecast_dates = pd.date_range(start=last_date, periods=num_forecasts + 1)[1:]  # Modified this line
    forecasts_series = pd.Series(data=forecast_values, index=forecast_dates, name="predicted_mean")
    
    return forecasts_series
```


```python
ser = load_data()
p, q, num_forecasts = 7, 7, 20

forecasts = arma_log_ret(ser, p, q, num_forecasts)
actual = pd.read_pickle("C:/Users/shincho/OneDrive - KLA Corporation/Desktop/Portfolio/Time_Series2/actual.pkl")  
rmse = np.sqrt(np.mean((actual - forecasts) ** 2))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ser, label="Original")
ax.plot(pd.concat([ser[-1:], forecasts]), label="Forecasted")
ax.plot(pd.concat([ser[-1:], actual]), label="Actual")

ax.set_xlabel("Day")
ax.set_title("Daily New COVID-19 Cases Worldwide\n" + f"A {len(forecasts)}-day Forecast, RMSE = {rmse:.2f}")
ax.legend()
plt.show()
```


    
![output_17_0](https://github.com/user-attachments/assets/866c5f52-10ae-4f01-8249-c2c3f55ad2f3)

    


## Part 5: ARMA on First-order Differences
First-order differencing helps in removing trends and seasonality from the series, making it more stationary. This step is crucial for improving the accuracy and robustness of the ARMA model. Hence, we fit an ARMA model on the first-order differences of the series and used this model to make forecasts


```python
def arma_first_diff(ser,p,q,num_forecasts):
    
    first_diff = ser.diff().dropna()
    model = ARIMA(first_diff, order =(p,0,q))
    model_fit = model.fit()
    forecast_diff = model_fit.forecast(steps=num_forecasts)
    last_actual = ser.iloc[-1]
    forecast_values = last_actual + np.cumsum(forecast_diff)
    last_date = ser.index[-1]
    forecast_dates = pd.date_range(start=last_date, periods = num_forecasts + 1)[1:]
    forecast_series = pd.Series(data=forecast_values, index=forecast_dates, name="predicted_mean")

    return forecast_series
```


```python
ser = load_data()
p, q, num_forecasts = 7, 7, 20

forecasts = arma_first_diff(ser, p, q, num_forecasts)
actual = pd.read_pickle("C:/Users/shincho/OneDrive - KLA Corporation/Desktop/Portfolio/Time_Series2/actual.pkl")  
rmse = np.sqrt(np.mean((actual - forecasts) ** 2))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ser, label="Original")
ax.plot(pd.concat([ser[-1:], forecasts]), label="Forecasted")
ax.plot(pd.concat([ser[-1:], actual]), label="Actual")

ax.set_xlabel("Day")
ax.set_title("Daily New COVID-19 Cases Worldwide\n" + f"A {len(forecasts)}-day Forecast, RMSE = {rmse: .2f}")
ax.legend()

del fig, ax, ser, p, q, num_forecasts, forecasts, actual
```


    
![output_20_0](https://github.com/user-attachments/assets/b6efce48-d6c8-45cf-b3cf-a988c2b16e56)

    


## Part 6: Multiple Time Series Forecasting
We extend our analysis to multiple time series by analyzing daily new COVID-19 cases from the top 5 countries with the highest cumulative cases using VAR and AR models 

Multiple time series analysis allows us to understand the interaction between different time series. Accurate forecasting in this context can help estimate the spread of the virus in several countries, thereby aiding in global health resource allocation and policy-making


```python
def load_data2():
    
    df = pd.read_csv("C:/Users/shincho/OneDrive - KLA Corporation/Desktop/Portfolio/Time_Series2/time_series_covid19_confirmed_global.csv")
    country = df.groupby("Country/Region").sum().iloc[:,2:]
    top = country.iloc[:,-1].sort_values(ascending=False).head(5)
    top_names = top.index.tolist()
    top_cases = country.loc[top_names]
    daily_new_cases = top_cases.diff(axis=1).iloc[:,1:].astype(float)
    daily_new_cases = daily_new_cases.T
    daily_new_cases.index = pd.to_datetime(daily_new_cases.index)
    daily_new_cases = daily_new_cases.drop(pd.to_datetime('2020-01-22'))
    
    return daily_new_cases

load_data2()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Country/Region</th>
      <th>US</th>
      <th>Brazil</th>
      <th>India</th>
      <th>Russia</th>
      <th>South Africa</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-23</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2020-01-24</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2020-01-25</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2020-01-26</th>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2020-01-27</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2020-08-17</th>
      <td>35112.0</td>
      <td>19373.0</td>
      <td>55018.0</td>
      <td>4839.0</td>
      <td>2541.0</td>
    </tr>
    <tr>
      <th>2020-08-18</th>
      <td>44091.0</td>
      <td>47784.0</td>
      <td>64572.0</td>
      <td>4718.0</td>
      <td>2258.0</td>
    </tr>
    <tr>
      <th>2020-08-19</th>
      <td>47408.0</td>
      <td>49298.0</td>
      <td>69672.0</td>
      <td>4790.0</td>
      <td>3916.0</td>
    </tr>
    <tr>
      <th>2020-08-20</th>
      <td>44023.0</td>
      <td>45323.0</td>
      <td>68900.0</td>
      <td>4767.0</td>
      <td>3880.0</td>
    </tr>
    <tr>
      <th>2020-08-21</th>
      <td>48693.0</td>
      <td>30355.0</td>
      <td>69876.0</td>
      <td>4838.0</td>
      <td>3398.0</td>
    </tr>
  </tbody>
</table>
<p>212 rows × 5 columns</p>
</div>



### 6a. Vector Autoregression (VAR)
VAR (Vector Autoregression) model captures the linear interdependencies among multiple time series. By forecasting several time series simultaneously, VAR models consider the influence of other series on eaech series, potentially improving the forecast accuracy.


```python
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.var_model import VARResults, VARResultsWrapper

def var_first_diff(df, p, num_forecasts):
    
    first_diff = df.diff().dropna()
    var_model = VAR(first_diff)
    var_res = var_model.fit(p)
    diff_forecasts = var_res.forecast(first_diff.values[-p:], num_forecasts)
    last_observed = df.iloc[-1, :].values
    actual_forecasts = [last_observed + diff_forecasts[0]]
    for i in range(1, num_forecasts):
        actual_forecasts.append(actual_forecasts[-1] + diff_forecasts[i])
    forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=num_forecasts, freq='D')
    forecasts = pd.DataFrame(actual_forecasts, index=forecast_index, columns=df.columns)
    
    return var_res, forecasts
```


```python
p, num_forecasts = 7, 20

stu_df = load_data2()
_, forecasts = var_first_diff(stu_df, p, num_forecasts)
actual = pd.read_pickle("C:/Users/shincho/OneDrive - KLA Corporation/Desktop/Portfolio/Time_Series2/actual_multi.pkl")
rmse = np.sqrt(np.mean((actual - forecasts) ** 2, axis=0)).round(2)

fig, axes = plt.subplots(1, 2, figsize=(25, 8), sharey=True, gridspec_kw={"wspace": 0})

stu_df.plot(ax=axes[0])

pd.concat([stu_df.iloc[-1:], actual]).plot(ax=axes[1], legend=False)
axes[1].set_prop_cycle(None)

pd.concat([stu_df.iloc[-1:], forecasts]).plot(ax=axes[1], legend=False, style=["-."] * stu_df.shape[1], linewidth=4)

axes[0].set_title(f"Daily New COVID-19 Cases until {forecasts.index[0]}", fontsize=14)
axes[1].set_title(r"Forecasted $-\cdot-\cdot-$ and Actual --- Daily New COVID-19 Cases" + "\n" + f"RMSE: {rmse.to_dict()}", fontsize=14)
del fig, axes, stu_df, p, num_forecasts, forecasts, actual, rmse
```


    
![output_25_0](https://github.com/user-attachments/assets/e098bd52-3735-4d2e-9f5c-e0522919a306)

    


### 6b. Independent AR Models for Each Time Series
Comparing VAR and independent AR models helps in identifying whether including interactions between countries improves forecasting accuracy. It provides insights into whether the countries influence each other's COVID-19 case trajectories


```python
def arma_first_diff(ser, p, q, num_forecasts):
    
    first_diff = ser.diff().dropna()
    arma_model = ARIMA(first_diff, order=(p, 0, q))
    arma_res = arma_model.fit()
    diff_forecasts = arma_res.forecast(steps=num_forecasts)
    last_observed = ser.iloc[-1]
    forecasts = last_observed + diff_forecasts.cumsum()

    return forecasts

def ar_first_diff(df, p, num_forecasts):
    
    forecast_df = pd.DataFrame(index=pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=num_forecasts))
    for country in df.columns:
        forecasts = arma_first_diff(df[country], p, 0, num_forecasts)
        forecast_df[country] = forecasts.values
    
    return forecast_df
```


```python
p, num_forecasts = 7, 20

stu_df = load_data2()
forecasts = ar_first_diff(stu_df, p, num_forecasts)
actual = pd.read_pickle("C:/Users/shincho/OneDrive - KLA Corporation/Desktop/Portfolio/Time_Series2/actual_multi.pkl")
rmse = np.sqrt(np.mean((actual - forecasts) ** 2, axis=0)).round(2)

fig, axes = plt.subplots(1, 2, figsize=(25, 8), sharey=True, gridspec_kw={"wspace": 0})

stu_df.plot(ax=axes[0])

pd.concat([stu_df.iloc[-1:], actual]).plot(ax=axes[1], legend=False)
axes[1].set_prop_cycle(None)

pd.concat([stu_df.iloc[-1:], forecasts]).plot(ax=axes[1], legend=False, style=["-."] * stu_df.shape[-1], linewidth=4)

axes[0].set_title(f"Daily New COVID-19 Cases until {forecasts.index[0]}", fontsize=14)
axes[1].set_title(r"Forecasted $-\cdot-\cdot-$ and Actual --- Daily New COVID-19 Cases" + "\n" + f"RMSE: {rmse.to_dict()}", fontsize=14)

del fig, axes, stu_df, p, num_forecasts, forecasts, actual, rmse
```


    
![output_28_0](https://github.com/user-attachments/assets/50b2d293-c41d-448c-952e-0005ae696efb)

    


## Part 7. Granger Causality
Granger Causality tests examine if one time series can predict another time series better than the latter's own past values. If including past values of another series improves the forecast, this suggests a causal influence


```python
def test_granger(df, p):
    first_diff = df.diff().dropna()
    var_model = VAR(first_diff)
    var_res = var_model.fit(p)
    countries = df.columns
    granger_df = pd.DataFrame(index=countries, columns=countries, dtype=float)
    to_test = [(caused, causing) for caused in countries for causing in countries if caused != causing]
    for caused, causing in to_test:
        test_result = var_res.test_causality(caused, causing, kind='f')
        granger_df.loc[caused, causing] = float(test_result.pvalue)
    granger_df = granger_df.astype(float)
    return granger_df
```


```python
stu_df, p = load_data2(), 7
stu_ans = test_granger(stu_df, 7)
caul_mtrx = stu_ans.rename(index={item: f"{item} caused by" for item in stu_ans.index})
caul_mtrx.where(caul_mtrx.isna(), caul_mtrx <= 0.01)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Country/Region</th>
      <th>US</th>
      <th>Brazil</th>
      <th>India</th>
      <th>Russia</th>
      <th>South Africa</th>
    </tr>
    <tr>
      <th>Country/Region</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>US caused by</th>
      <td>NaN</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>Brazil caused by</th>
      <td>False</td>
      <td>NaN</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>India caused by</th>
      <td>False</td>
      <td>True</td>
      <td>NaN</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>Russia caused by</th>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>South Africa caused by</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



**Noticeable Results:**

**US Causality**: The matrix suggests Brazil has a predictive capability over the US which isn't immediately intuitive as one would think larger connections might be driving factors

**Russia's Dependency of the U.S**: This emphasizes a uniquely strong lionk indicating specific international dynamics or policies during the epidemic

**South Africa's Isolation**: Counter-intuitive as it acts independently signaling localized control or varies epidemic drivers


![image](https://github.com/user-attachments/assets/3a59b4b4-a774-4cb2-bd1d-d8e0749f7a5f)

