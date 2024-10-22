---
layout: single
title:  "Daily COVID-19 Cases"
subtitle: "Time Series"
categories: python
tag: [Time Series, Trend Curve, WMA, EMA, Euclidean, Cosine Similarity, DTW]
toc: true
---

## 2024 Analyzing Daily New COVID-19 Cases

## Part 1: Global COVID Cases

### Resources
- [Johns Hopkins University CSSE COVID-19](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series)

### Background
In this project, we aim to analyze the time series of daily new COVID-19 cases using data from Johns Hopkins University CSSE COVID-19 dataset. We will use various time series analysis techniques to discern patterns in the data

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

df = pd.read_csv("C:/Users/shincho/OneDrive - KLA Corporation/Desktop/Portfolio/Time_Series1/Global_COVID19_Cases.csv")
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
    daily_new_cases = df.iloc[:,4:].sum(axis=0)
    daily_new_cases.index = pd.to_datetime(daily_new_cases.index)
    daily_new_cases = daily_new_cases.diff().iloc[1:]

    return daily_new_cases

load_data()
```

    C:\Users\shincho\AppData\Local\Temp\ipykernel_30444\2334956022.py:3: UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.
      daily_new_cases.index = pd.to_datetime(daily_new_cases.index)
    




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




```python
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(load_data())
ax.set_xlabel("Day")
ax.set_ylabel("# Cases")
ax.set_title("Daily New COVID-19 Cases Worldwide")
plt.show()

del fig, ax
```

    C:\Users\shincho\AppData\Local\Temp\ipykernel_30444\2334956022.py:3: UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.
      daily_new_cases.index = pd.to_datetime(daily_new_cases.index)
    


    
![output_3_1](https://github.com/user-attachments/assets/3c851702-d870-4797-9ad1-34f3a0bbfab9)

    


### 1. Performing Seasonal Decomposition
Next, we decompose the time series into trend, seasonal, and residual components to understand the underlying patterns


```python
from statsmodels.tsa.seasonal import seasonal_decompose, DecomposeResult

# We generate a function to perform seasonal decomposition
def sea_decomp(ser, model = "additive"):
    result = seasonal_decompose(ser,model=model)
    return result

# Visualize the seasonal decomposition
fig, axes = plt.subplots(4,1,figsize=(10,6),sharex=True)
res = sea_decomp(load_data(), model="additive")

axes[0].set_title("Additive Seasonal Decomposition")
axes[0].plot(res.observed)
axes[0].set_ylabel("Observed")

axes[1].plot(res.trend)
axes[1].set_ylabel("Trend")

axes[2].plot(res.seasonal)
axes[2].set_ylabel("Seasonal")

axes[3].plot(res.resid)
axes[3].set_ylabel("Residual")

axes[3].set_xlabel("Day")
fig.suptitle("Daily New COVID-19 Cases Worldwide", x=0.513, y=0.95)
plt.show()

```

    C:\Users\shincho\AppData\Local\Temp\ipykernel_30444\2334956022.py:3: UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.
      daily_new_cases.index = pd.to_datetime(daily_new_cases.index)
    


    
![output_5_1](https://github.com/user-attachments/assets/e3582c80-7190-40bb-bd32-6436fe64c749)

    


**Interpretation**
- Observed Component: The original series of daily new cases
- Trend Component: The overall direction (upward or downward) in which the series is moving
- Seasonal component: Regular patterns in the Corona data that occur at periodic intervals
- Residual Component: Remaining noisee after the trend and seasonal components are removed

### 2. Fitting a Trend Curve
We fit an n-th order polynomial to the time eries to uncover non-linear trends


```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def fit_trend(ser, n):
    train_X, train_y = np.arange(len(ser)), ser.values
    train_X = PolynomialFeatures(n).fit_transform(train_X.reshape(-1, 1))
    lin_reg = LinearRegression().fit(train_X, train_y)

    trend_curve = lin_reg.predict(train_X)

    return trend_curve

fig, ax = plt.subplots(figsize=(10,6))
ser = load_data()
preds = fit_trend(ser,10)
ax.plot(ser.index, ser.values, label = "Original")
ax.plot(ser.index, preds, label = "Fitted trend curve")
ax.set_xlabel("Day")
ax.set_ylabel("# Cases")
ax.set_title("Daily New COVID-19 Cases Worldwide")
ax.legend()
plt.show()
```

    C:\Users\shincho\AppData\Local\Temp\ipykernel_30444\2334956022.py:3: UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.
      daily_new_cases.index = pd.to_datetime(daily_new_cases.index)
    


    
![output_7_1](https://github.com/user-attachments/assets/a2c3bafe-a80b-4e10-8cbb-c709d2c2291a)

    


**Interpretation**
- The trend curve provides a smoothed representation of the underlying trend in hte data. Peaks and thoughts in the trend curve indicate significant changes in the rate of new cases

### 3. Calculating Weighted Moving Average (WMA)
We calculate the weighted moving average to smooth out short-term fluctuations and highlight trends


```python
def calc_wma(ser, wd_size, weights =1):
    if isinstance(weights, int):
        weights = np.full(wd_size, weights, dtype=float)
    wma = np.zeros(len(ser))
    for i in range(len(ser)):
        if i >= wd_size:
            window = ser[i - wd_size + 1: i+1]
            wma[i] = np.sum(weights * window) / np.sum(weights)
        else:
            window = ser[:i+1]
            partial = weights[-(i+1):]
            wma[i] = np.sum(partial * window) / np.sum(partial)
    return wma

wd_size = 7
weights = np.arange(1, wd_size + 1)
ser = load_data()
wma = calc_wma(ser, wd_size, weights = weights)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ser.index, ser.values, label = "Original")
ax.plot(ser.index, wma, label = "WMA")
ax.set_xlabel("Day")
ax.set_ylabel("# Cases")
ax.set_title("Daily New COVID-19 Cases Worldwide")
ax.legend()
plt.show()
```

    C:\Users\shincho\AppData\Local\Temp\ipykernel_30444\2334956022.py:3: UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.
      daily_new_cases.index = pd.to_datetime(daily_new_cases.index)
    


    
![output_9_1](https://github.com/user-attachments/assets/f01a143d-8515-416d-b604-3260c1fb3c83)

    


**Interpretation**
- The WMA smooths out short-term fluctuations, making it easier to see long-term trends. Heavier weights on more recent observations help in identifying current trends

### 4. Calculating Exponential Moving Average (EMA)
We calculate the exponential moving average to give more weight to recent observations


```python
def calc_time_ema(ser, lmbd=0.0):
    time_ema = []
    if lmbd == 0:
        cumulative = 0.0
        for i in range(len(ser)):
            cumulative += ser[i]
            time_ema.append(cumulative / (i + 1))
    else:
        for i in range(len(ser)):
            numerator = 0.0
            denominator = 0.0
            for j in range(i + 1):
                weight = np.exp(-lmbd * (i - j))
                numerator += weight * ser[j]
                denominator += weight
            time_ema.append(numerator / denominator)
    return np.array(time_ema)

ser = load_data()
ema = calc_time_ema(ser, lmbd=0.5)
fig, ax = plt.subplots(figsize = (10,6))
ax.plot(ser.index, ser.to_numpy(), label = "Original")
ax.plot(ser.index, ema, label = "Time EMA")
ax.set_xlabel("Day")
ax.set_ylabel("# Cases")
ax.set_title("Daily New COVID-19 Cases Worldwide")
ax.legend()
plt.show()
```

    C:\Users\shincho\AppData\Local\Temp\ipykernel_30444\2334956022.py:3: UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.
      daily_new_cases.index = pd.to_datetime(daily_new_cases.index)
    C:\Users\shincho\AppData\Local\Temp\ipykernel_30444\7130710.py:14: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      numerator += weight * ser[j]
    


    
![output_11_1](https://github.com/user-attachments/assets/614d00c7-dcca-48f8-96ce-ce7fec78eb77)

    


**Interpretation**
- The EMA emphasizes more recent observations, making it sensitive to short-term changes. This can be useful for identifying emerging trends or shifts

## Part 2: Top 5 Country COVID Cases

### 5. Data Analysis for Top 5 Countries
We load and preprocess the data to extract daily new cases for the top 5 countries with the most cumulative cases as of August 21, 2020


```python
def load_data():

    country = df.groupby("Country/Region").sum().iloc[:,2:]
    top = country.iloc[:,-1].sort_values(ascending=False).head(5)
    top_names = top.index.tolist()

    top_cases = country.loc[top_names]
    daily_new_cases = top_cases.diff(axis=1).iloc[:,1:].astype(float)

    daily_new_cases = daily_new_cases.T
    daily_new_cases.index = pd.to_datetime(daily_new_cases.index)

    return daily_new_cases

axes = load_data().plot(figsize=(10,6), title = "Daily New COVID-19 Cases", ylabel="# Cases")
plt.show()
```

    C:\Users\shincho\AppData\Local\Temp\ipykernel_30444\1019473538.py:11: UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.
      daily_new_cases.index = pd.to_datetime(daily_new_cases.index)
    


    
![output_13_1](https://github.com/user-attachments/assets/2931bf9d-4e5c-41d6-97b3-3ddee22ef16a)

    


**Interpretation**
- This allows us to compare trends between the top 5 most affected countries. Peaks can indicate significant outbreaks or improvements in reporting

### 6. Extracting Seasonal Componenets
We extract seasonal components from the time series for each of the top 5 countries


```python
from statsmodels.tsa.seasonal import seasonal_decompose

def sea_decomp(df, model="additive"):
    sea_df = {}
    
    for x in df.columns:
        result = seasonal_decompose(df[x], model = model)
        sea_df[x] = result.seasonal
    sea_df = pd.DataFrame(sea_df, index = df.index)
    
    return sea_df

df = load_data()
axes = sea_decomp(df).plot(figsize=(10, 6), title="Seasonal Component of Daily New COVID-19 Cases")
plt.show()
```

    C:\Users\shincho\AppData\Local\Temp\ipykernel_30444\1019473538.py:11: UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.
      daily_new_cases.index = pd.to_datetime(daily_new_cases.index)
    


    
![output_15_1](https://github.com/user-attachments/assets/ce2c8d50-1883-4dd0-b36b-98bdceb13e44)

    


**Interpretation**
- By decomposing the data into seasonal components, we can identify repetitive patterns or cycles, such as weekly or monthly trends in the number of cases

### 7. Calculating Euclidean Distance
We compute pairwise Euclidean Distance among the seasonal patterns of the time series


```python
def calc_euclidean_dist(df):
    country = df.columns
    euclidean_dist_df = pd.DataFrame(index = country, columns = country, dtype = float)

    for x in country:
        for y in country:
            if x == y:
                euclidean_dist_df.loc[x,y] =0.0
            else:
                euclidean_dist_df.loc[x,y] = np.sqrt(np.sum((df[x] - df[y]) ** 2))
    return euclidean_dist_df

calc_euclidean_dist(sea_decomp(df))
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
      <th>US</th>
      <th>Brazil</th>
      <th>India</th>
      <th>Russia</th>
      <th>South Africa</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>US</th>
      <td>0.000000</td>
      <td>37820.182820</td>
      <td>27033.765938</td>
      <td>33980.853042</td>
      <td>30084.611964</td>
    </tr>
    <tr>
      <th>Brazil</th>
      <td>37820.182820</td>
      <td>0.000000</td>
      <td>57731.118383</td>
      <td>63807.259063</td>
      <td>60982.660267</td>
    </tr>
    <tr>
      <th>India</th>
      <td>27033.765938</td>
      <td>57731.118383</td>
      <td>0.000000</td>
      <td>9075.172246</td>
      <td>4483.138210</td>
    </tr>
    <tr>
      <th>Russia</th>
      <td>33980.853042</td>
      <td>63807.259063</td>
      <td>9075.172246</td>
      <td>0.000000</td>
      <td>5640.047894</td>
    </tr>
    <tr>
      <th>South Africa</th>
      <td>30084.611964</td>
      <td>60982.660267</td>
      <td>4483.138210</td>
      <td>5640.047894</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



**Interpretation**
The Euclidean Distance matrix provides insights into the similarity between seasonal patterns of different countries. Smaller distances indicate more similar patterns

**Summary of Observations**
- India and South Africa have the smallest Euclidean Distance (4490.020448), suggesting their seasonal patterns are highly similar
- Brazil shows substantial dissimilarity with other country, having larger distances overall, especially with Russia (63663.896821)

### 8. Calculating Cosine Similarity
We compute pairwise Cosine Similarity among the seasonal patterns of the time series


```python
def calc_cos_sim(df):
    country = df.columns
    cos_sim_df = pd.DataFrame(index = country, columns = country, dtype = float)
    
    for x in country:
        for y in country:
            if x == y:
                cos_sim_df.loc[x,y] = 1.0
            else:
                norm_x = np.linalg.norm(df[x])
                norm_y = np.linalg.norm(df[y])
                if norm_x == 0 or norm_y == 0:
                    cos_sim_df.loc[x,y] = 0.0
                else:
                    cos_sim_df.loc[x,y] = np.dot(df[x],df[y]) / (norm_x * norm_y)
    return cos_sim_df

calc_cos_sim(sea_decomp(df))
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
      <th>US</th>
      <th>Brazil</th>
      <th>India</th>
      <th>Russia</th>
      <th>South Africa</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>US</th>
      <td>1.000000</td>
      <td>0.867477</td>
      <td>0.785003</td>
      <td>-0.325869</td>
      <td>0.665168</td>
    </tr>
    <tr>
      <th>Brazil</th>
      <td>0.867477</td>
      <td>1.000000</td>
      <td>0.633244</td>
      <td>-0.633881</td>
      <td>0.403142</td>
    </tr>
    <tr>
      <th>India</th>
      <td>0.785003</td>
      <td>0.633244</td>
      <td>1.000000</td>
      <td>0.086160</td>
      <td>0.916845</td>
    </tr>
    <tr>
      <th>Russia</th>
      <td>-0.325869</td>
      <td>-0.633881</td>
      <td>0.086160</td>
      <td>1.000000</td>
      <td>0.168506</td>
    </tr>
    <tr>
      <th>South Africa</th>
      <td>0.665168</td>
      <td>0.403142</td>
      <td>0.916845</td>
      <td>0.168506</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



**Interpretation**
- The Cosine Similarity matrix shows how similar the seasonal patterns are in terms of their direction. Values close to 1 indicate high similarity, while values close to -1 indicate dissimilarity

**Summary of Observations**
- South Africa and India have the highest Cosine Similarity (0.916845), indicating highly aligned seasonal patterns
- Russia shows negative similarity with the US and Brazil, indicating their paterns are largely dissimilar in direction

### 9. Calculating Dynamic Time Warping (DTW) Cost
#### 9.a Pairwise DTW Cost Calculation
We calculate the DTW cost to measure alignment cost between two time series


```python
import math

def calc_pairwise_dtw_cost(x,y,ret_matrix=False):
    len_x = len(x)
    len_y = len(y)
    cost_matrix = np.zeros((len(y),len(x)))

    dist_fn = lambda a,b: (a-b) ** 2
    cost_matrix[0,0] = dist_fn(x[0],y[0])

    for i in range(1,len_x):
        cost_matrix[0,i] = dist_fn(x[i], y[0]) + cost_matrix[0, i-1]
    
    for i in range(1,len_y):
        cost_matrix[i,0] = dist_fn(x[0],y[i]) + cost_matrix[i-1, 0]

    for i in range(1, len_y):
        for j in range(1, len_x):
            cost = dist_fn(x[j], y[i])
            cost_matrix[i,j] = cost + min(
                cost_matrix[i-1,j],
                cost_matrix[i,j-1],
                cost_matrix[i-1,j-1]
            )
    dtw_cost = cost_matrix[len_y - 1, len_x - 1]

    return cost_matrix if ret_matrix else dtw_cost
```

#### 9.b All pairwise DTW Costs Calculation
We calculat ethe DTW costs for all pairs of countries


```python
def calc_dtw_cost(df):
    country = df.columns
    dtw_cost_df = pd.DataFrame(index = country, columns = country, dtype = float)

    for x in country:
        for y in country:
            if x == y:
                dtw_cost_df.loc[x,y] = 0.0
            else:
                dtw_cost_df.loc[x,y] = calc_pairwise_dtw_cost(df[x],df[y])
    return dtw_cost_df

np.sqrt(calc_dtw_cost(sea_decomp(df)))
```

    C:\Users\shincho\AppData\Local\Temp\ipykernel_30444\1590672362.py:9: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      cost_matrix[0,0] = dist_fn(x[0],y[0])
    C:\Users\shincho\AppData\Local\Temp\ipykernel_30444\1590672362.py:12: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      cost_matrix[0,i] = dist_fn(x[i], y[0]) + cost_matrix[0, i-1]
    C:\Users\shincho\AppData\Local\Temp\ipykernel_30444\1590672362.py:15: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      cost_matrix[i,0] = dist_fn(x[0],y[i]) + cost_matrix[i-1, 0]
    C:\Users\shincho\AppData\Local\Temp\ipykernel_30444\1590672362.py:19: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      cost = dist_fn(x[j], y[i])
    




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
      <th>US</th>
      <th>Brazil</th>
      <th>India</th>
      <th>Russia</th>
      <th>South Africa</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>US</th>
      <td>0.000000</td>
      <td>32117.911435</td>
      <td>23540.021106</td>
      <td>32281.607600</td>
      <td>28070.379461</td>
    </tr>
    <tr>
      <th>Brazil</th>
      <td>32117.911435</td>
      <td>0.000000</td>
      <td>53554.525689</td>
      <td>62011.469685</td>
      <td>57300.471943</td>
    </tr>
    <tr>
      <th>India</th>
      <td>23540.021106</td>
      <td>53554.525689</td>
      <td>0.000000</td>
      <td>7636.616154</td>
      <td>4477.393461</td>
    </tr>
    <tr>
      <th>Russia</th>
      <td>32281.607600</td>
      <td>62011.469685</td>
      <td>7636.616154</td>
      <td>0.000000</td>
      <td>4212.921781</td>
    </tr>
    <tr>
      <th>South Africa</th>
      <td>28070.379461</td>
      <td>57300.471943</td>
      <td>4477.393461</td>
      <td>4212.921781</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



**Interpretation**
- The DTW cost matrix provides a measure of alignment cost between time series. Smaller costs indicate more similar time series in terms of their shape and timing of patterns

**Summary of Observations**
- Similar to Euclidean findings, India and South Africa have a low DTW cost (4477.393461), indicating highly similar seasonal patterns
- Again, Russia and Brazil exhibit high DTW costs with other countries, reflecting greater dissimilarity

Vaccine shot - First Day vs. Second Day
![image](https://github.com/user-attachments/assets/8c4f15e3-e9ed-4474-94e4-22700c77272a)

