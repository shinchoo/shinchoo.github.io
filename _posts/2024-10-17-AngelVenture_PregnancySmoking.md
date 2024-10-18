---
layout: single
title:  "Angel Investor Impact on Saas Ventures & Smoking Rates in Pregnancy with Higher Cigarette Taxes"
subtitle: "Causal Inference 4: Difference in Differences"
categories: python
tag: [Causal Inference, Difference-in-Differences]
toc: true
---

## Analysis of Angel Investor Impact on SaaS Ventures and Smoking Rates in Pregnancy with Higher Cigarette Taxes

### Part 1:Impact of Angel Investor Support SaaS Ventures

### Resources
- [Kerr and Learner, Review of Financial Studies 2014](https://www.jstor.org/stable/24464820)

### Background
Entrepreneurial finance often examines how early-stage financiers, like angel investors, influence the firms they support. This study uses data from a prominent SaaS investor angel group composed of 39 angel investors who play a crucial role in selecting ventures to support. Ventures with 20 or more angel investor votes to receive investment and advice.

### Terminology
    - **SaaS (Software as a Service)**: This is a software distribution model in which applications are hosted by a service provider and made available to customers over the internet.
    - **Angel Investor**: These are individuals who provide capital for startups or early-stage companies, often in exchange for ownership equity or convertible debt
    - **Early-Stage Financier**: This term refers to investors who provide funding to startups and early-stage companies

### Data
The dataset contains data on several SaaS ventures:
- **dau_mau**: Daily active user to monthly active user ratio, measured 15 months after voting
- **revenue_g**: Revenue growth rate, measured 15 months after voting
- **vote**: Number of angel investor votes received by each venture


```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns

data_rd = pd.read_csv("C:/Users/shincho/OneDrive - KLA Corporation/Desktop/Portfolio/Causal_Inference4/RD.csv")
data_rd.head()
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
      <th>vote</th>
      <th>dau_mau</th>
      <th>revenue_g</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32</td>
      <td>0.824858</td>
      <td>1.371313</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0.030839</td>
      <td>0.488632</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15</td>
      <td>0.199545</td>
      <td>0.718289</td>
    </tr>
    <tr>
      <th>3</th>
      <td>28</td>
      <td>0.652112</td>
      <td>1.371974</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24</td>
      <td>0.602617</td>
      <td>1.107473</td>
    </tr>
  </tbody>
</table>
</div>



### 1. Scatter Plot of DAU/MAU Ratio vs. Vote


```python
plt.scatter(data_rd['vote'], data_rd['dau_mau'])
plt.xlabel('Vote')
plt.ylabel('DAU/MAU')
plt.title('Scatter plot of DAU_MAU Ratio vs. Vote')
plt.show()
```

![output_3_0](https://github.com/user-attachments/assets/959a9c51-eeea-48d0-ba8d-1c20149faf33)

    

    


- This scatterplot shows the relationship between vote and dau_mau is linear, yet the slope seems to differ across two sides of the cutoff

### 2. Linear Reglationship Analysis
We start by creating Binary and Re-Centered Variables.


```python
data_rd['D'] = np.where(data_rd['vote'] >= 20, 1, 0)
data_rd['vote_c'] = data_rd['vote'] - 20
data_rd.head()
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
      <th>vote</th>
      <th>dau_mau</th>
      <th>revenue_g</th>
      <th>D</th>
      <th>vote_c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32</td>
      <td>0.824858</td>
      <td>1.371313</td>
      <td>1</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0.030839</td>
      <td>0.488632</td>
      <td>0</td>
      <td>-20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15</td>
      <td>0.199545</td>
      <td>0.718289</td>
      <td>0</td>
      <td>-5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>28</td>
      <td>0.652112</td>
      <td>1.371974</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24</td>
      <td>0.602617</td>
      <td>1.107473</td>
      <td>1</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



We are now conducting Regression Analysis with Robust Standard Errors to measure the impact of angel group support (D) on the DAU/MAU ratio by accounting for the different slops on either side of the cutoff


```python
data_rd['vote_c_D'] = data_rd['vote_c'] * data_rd['D']
X = data_rd[['vote_c','D','vote_c_D']]
X = sm.add_constant(X)
y = data_rd['dau_mau']
model = sm.OLS(y,X).fit()
d_coeff = round(model.params['D'],4)
d_coeff
```




    0.3016



- This coefficient 0.3016 means that ventuers receiving angel group support (i.e., those with 20 or more votes) have a DAU/MAUI ratio that is, on average, 30.16% higher than those that did not receive the support

### 3. Scatter Plot of Revenue Growth vs. Vote


```python
plt.scatter(data_rd['vote_c'],data_rd['revenue_g'])
plt.xlabel('Vote')
plt.ylabel('Revenue Growth')
plt.title('Scatter plot of Revenue Growth vs. Vote')
plt.show()
```


    
![output_9_0](https://github.com/user-attachments/assets/03b85565-558e-450e-8982-328a7b211f41)

    


- The visual shows that the relationship between vote_c and revenue_g is non-linear

### 4. Non-Linear Relationship Analysis
We start by creating a Sqaured Variable


```python
data_rd['vote_c_sq'] = data_rd['vote_c'] ** 2
data_rd.head()
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
      <th>vote</th>
      <th>dau_mau</th>
      <th>revenue_g</th>
      <th>D</th>
      <th>vote_c</th>
      <th>vote_c_D</th>
      <th>vote_c_sq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32</td>
      <td>0.824858</td>
      <td>1.371313</td>
      <td>1</td>
      <td>12</td>
      <td>12</td>
      <td>144</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0.030839</td>
      <td>0.488632</td>
      <td>0</td>
      <td>-20</td>
      <td>0</td>
      <td>400</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15</td>
      <td>0.199545</td>
      <td>0.718289</td>
      <td>0</td>
      <td>-5</td>
      <td>0</td>
      <td>25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>28</td>
      <td>0.652112</td>
      <td>1.371974</td>
      <td>1</td>
      <td>8</td>
      <td>8</td>
      <td>64</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24</td>
      <td>0.602617</td>
      <td>1.107473</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>



Now, we conduct Regression Analysis with Non-Linear Term


```python
X2 = data_rd[['vote_c','vote_c_sq','D']]
X2 = sm.add_constant(X2)
y2 = data_rd['revenue_g']
model2 = sm.OLS(y2,X2).fit()
d_coeff2 = round(model2.params['D'],4)
d_coeff2
```




    0.2474



- The coefficient 0.2474 means that ventures receiving angel group support have a revenue growth rate taht is, on average, 24.74% higher than those who did not receive the support

### 5. Bandwidth-Specific Analysis

We conduct regression analysis with Bandwidth equal to 3 votes


```python
bandwidth_data = data_rd[(data_rd['vote'] >= 17) & (data_rd['vote'] <= 23)]
X3 = bandwidth_data[['vote_c','D']]
X3 = sm.add_constant(X3)
y3 = bandwidth_data['revenue_g']
model3 = sm.OLS(y3,X3).fit()
d_coeff3 = round(model3.params['D'],4)
d_coeff3
```




    0.2488





The coefficient 0.2488 is very close to the previous estimate of 0.2474, suggesting that our results are robust. Ventures that is receiving angel group support have a revenue growth rate that is, on average, 24.88% higher when considering votes close to the cutoff

### Part 2: Effect of Cigarette Taxes on Smoking Rates During Pregnancy

### Background
The US Surgeon General estimates that smoking during pregnancy doubles the chance of low birth weight in babies. To reduce smoking rates during pregnancy, higher cigarette taxes are suggested. On May 1, 1994, Michigan raised the tax from 25 to 75 cents per pack.

### Data
The dataset includes:
- **state**: 2-digit state FIPS code (Michigan: 26, Iowa: 19)
- **smoked**: Dummy variable indicating if the mother smoked during pregnancy
- **year**: Observation period (1: May 1, 1992 - April 30, 1993; 2: 2 May 1, 1993 - April 30, 1994; 3: may 1, 1994 - April 30, 1995)


```python
data_dd = pd.read_csv("C:/Users/shincho/OneDrive - KLA Corporation/Desktop/Portfolio/Causal_Inference4/DD.csv")
data_dd.head()
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
      <th>state</th>
      <th>smoked</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>19</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>19</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### 1. Visual Inspection of Parallel Trends


```python
smoking = data_dd[data_dd['year'].isin([1,2])].groupby(['state','year'])['smoked'].mean().reset_index()
michigan = smoking[smoking['state'] == 26]
iowa = smoking[smoking['state'] == 19]

plt.plot(michigan['year'], michigan['smoked'], marker='o',label='Michigan',color='blue')
plt.plot(iowa['year'],iowa['smoked'],marker='o',label='Iowa',color='green')

plt.ylim(0,0.4)
plt.yticks([0,0.2,0.4])
plt.xlim(1,3)
plt.xticks([1,2,3])
plt.xlabel('Year')
plt.ylabel('Smoking Rate')
plt.title('Pre treatment Smoking Rates in Michigan and Iowa')
plt.show()
```

![output_21_0](https://github.com/user-attachments/assets/3aff8c73-0059-4319-bf67-2b0be7ac845d)

    


- Based on the graph, the lines are approximately parallel, indicating that differences-in-differences method is a valid approach

### 2. Differences-in-Differences Analysis


```python
data_dd['treat'] = data_dd['state'].apply(lambda x: 1 if x == 26 else 0)
data_dd['post'] = data_dd['year'].apply(lambda x: 1 if x == 3 else 0)
data_dd.head()
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
      <th>state</th>
      <th>smoked</th>
      <th>year</th>
      <th>treat</th>
      <th>post</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>19</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>19</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Now we calculate means for differences-in-differences table


```python
cell_i = round(data_dd[(data_dd['treat'] == 1) & (data_dd['post'] == 0)]['smoked'].mean(), 4)
cell_ii = round(data_dd[(data_dd['treat'] == 1) & (data_dd['post'] == 1)]['smoked'].mean(), 4)
cell_iii = cell_ii - cell_i
cell_iv = round(data_dd[(data_dd['treat'] == 0) & (data_dd['post'] == 0)]['smoked'].mean(), 4)
cell_v = round(data_dd[(data_dd['treat'] == 0) & (data_dd['post'] == 1)]['smoked'].mean(), 4)
cell_vi = round(cell_v - cell_iv, 4)
cell_vii = round(cell_i - cell_iv, 4)
cell_viii = round(cell_ii - cell_v, 4)
cell_ix = round(cell_iii - cell_vi, 4)

cell_i, cell_ii, cell_iii, cell_iv, cell_v, cell_vi, cell_vii, cell_viii, cell_ix
```




    (0.1923, 0.1396, -0.0527, 0.1856, 0.1639, -0.0217, 0.0067, -0.0243, -0.031)



|                               | Treatment (Michigan) | Control (Iowa)       | Difference (Treatment-Control) |
| ----------------------------- | -------------------- | -------------------- | ------------------------------ |
| Pre-treatment (year1 & year2) | 0.1923               | 0.1856               | 0.0067                         |
| Post-treatment (year 3)       | 0.1396               | 0.1639               | -0.0243                        |
| Difference (Post - Pre)       | -0.0527              | -0.0217              | -0.031                         |

- The difference-in-differences estimate of -0.031 suggests that the tax increase led to a 3.1 percentage point reduction in smoking rates among pregnant women in Michigan compared to the control group (Iowa)

### 3. Regression Analysis for Differences-in-Differences


```python
data_dd['treat_post'] = data_dd['treat'] * data_dd['post']
X4 = data_dd[['treat','post','treat_post']]
X4 = sm.add_constant(X4)
y4 = data_dd['smoked']
model4 = sm.OLS(y4, X4).fit()
interact_coeff = round(model4.params['treat_post'],4)
interact_coeff
```




    -0.031



- The coefficient -0.031 means that the cigarette tax increase in Michigan results in a 3.1 percentage point reduction in smoking rates among pregnant women

![image](https://github.com/user-attachments/assets/73e9e717-9ac4-422b-8276-dadc0c7d2e4a)
