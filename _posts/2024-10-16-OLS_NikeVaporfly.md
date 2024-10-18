---
layout: single
title:  "Nike Vaporfly Shoes"
subtitle: "Causal Inference 2: OLS"
categories: python
tag: [Causal Inference, OLS]
---

## Nike Vaporfly Shoes Analysis

### Resources
- ["Nike says its $250 running shoes will make you run much faster. What if that's actually true?"](https://www.nytimes.com/interactive/2018/07/18/upshot/nike-vaporfly-shoe-strava.html)

### Background
Nike claims that its $250 running shoes called "Vaporfly" will make you run much faster. This project explores the accuracy of this claim using data from marathon runners

### Part 1: Data Transformation and Analysis

### Data
The marathon dataset contains 5 variables for 24,699 runners:
- **age**: age of runner (min value: 18, max value: 55)
- **male**: dummy variable for gender, equal to 0 if female, 1 if male
- **marathoner_type**:

    - **seasoned**: if runner has at least 3 prior completed marathons,

    - **enthusiastic**: if runner has completed 1 or 2 prior completed marathons,
    
    - **first_timer**: if this is a runner's first time running a marathon
    
- **vaporfly**: if a runner's racing shoes is Nike Vaporfly, 0 otherwise
- **race_time**: marathon completion time in seconds


```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from causalinference import CausalModel

data_marathon = pd.read_csv('C:/Users/shincho/OneDrive - KLA Corporation/Desktop/Portfolio/Causal_Inference2/marathon.csv')
data_marathon.head()
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
      <th>age</th>
      <th>marathoner_type</th>
      <th>vaporfly</th>
      <th>race_time</th>
      <th>male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41</td>
      <td>enthusiastic</td>
      <td>1</td>
      <td>11755.176</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42</td>
      <td>enthusiastic</td>
      <td>1</td>
      <td>14980.950</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>39</td>
      <td>enthusiastic</td>
      <td>0</td>
      <td>12342.542</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>29</td>
      <td>enthusiastic</td>
      <td>0</td>
      <td>13142.107</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>34</td>
      <td>enthusiastic</td>
      <td>1</td>
      <td>13255.874</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### 1. Log Transformation of Race Time
We transform the 'race_time' variable by taking its natural log to create a new variable called 'ln_race_time'


```python
data_marathon['ln_race_time'] = np.log(data_marathon['race_time'])
data_marathon.sample()
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
      <th>age</th>
      <th>marathoner_type</th>
      <th>vaporfly</th>
      <th>race_time</th>
      <th>male</th>
      <th>ln_race_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2960</th>
      <td>31</td>
      <td>seasoned</td>
      <td>0</td>
      <td>14619.159</td>
      <td>0</td>
      <td>9.590088</td>
    </tr>
  </tbody>
</table>
</div>



### 2. Means of ln_race_time
We compute the mean of ln_race_time for runners using and not using Vaporfly shoes


```python
mean_vaporfly = data_marathon[data_marathon['vaporfly'] == 1]['ln_race_time'].mean()
mean_non_vaporfly = data_marathon[data_marathon['vaporfly'] == 0]['ln_race_time'].mean()
mean_diff1 = mean_vaporfly - mean_non_vaporfly
mean_diff1 = round(mean_diff1, 4)
mean_diff1
```




    -0.064



- **Interpretation**: The mean difference in the natural logarithm of race times (ln_race_time) between runners who wore Vaporfly shoes and those who did not is approximately -0.064. This suggests that, on average, runners who wore Vaporfly shoes had a decrease in ln_race_time compared to those who did not. The negative sign indicates that the Vaporfly shoes are associated with faster race time.

### 3. Average Treatment Effects (ATE) Using Nearest Neighbor Matching
We use the CausalModel for ATE estimation


```python
data_sample = data_marathon.sample(n = 2000, random_state = 123)
treatment = data_sample['vaporfly'].values
outcome = data_sample['ln_race_time'].values.reshape((-1,1))
features = data_sample[['age']].values

causal = CausalModel(Y = outcome, D = treatment, X = features)
causal.est_via_matching()
ate = causal.estimates['matching']['ate']
ate = round(ate,4)
ate = float(ate)
ate
```




    -0.0372



- **Interpretation**: Using nearest neighbor matching on the variable age, the estimated Average Treatment Effect (ATE) of wearing Nike Vaporfly shoes on ln_race_time is -0.0372. This further supports the indicates that Vaporfly shoes contribute to a reduction in race times, though the effect size is smaller than the mean difference observed without matching.

### 4. Propensity Score Matching
Propensity score matching (PSM) is used to control for potential confounding variables by matching treated units with control units that have similar propensity scores.
Hence, we create binary variables then estimate ATE using propensity score matching by calculating propensity score and performing matching.


```python
data_marathon['seasoned'] = np.where(data_marathon['marathoner_type'] == 'seasoned',1,0)
data_marathon['enthusiastic'] = np.where(data_marathon['marathoner_type'] == 'enthusiastic',1,0)

# Logistic regression model to calculate propensity score
logit_model = sm.Logit(data_marathon['vaporfly'],data_marathon[['age','male','seasoned','enthusiastic']])
result = logit_model.fit()
data_marathon['propensity_score'] = result.predict()
data_marathon.head()
```

    Optimization terminated successfully.
             Current function value: 0.690706
             Iterations 4
    




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
      <th>age</th>
      <th>marathoner_type</th>
      <th>vaporfly</th>
      <th>race_time</th>
      <th>male</th>
      <th>ln_race_time</th>
      <th>seasoned</th>
      <th>enthusiastic</th>
      <th>propensity_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41</td>
      <td>enthusiastic</td>
      <td>1</td>
      <td>11755.176</td>
      <td>1</td>
      <td>9.372049</td>
      <td>0</td>
      <td>1</td>
      <td>0.482016</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42</td>
      <td>enthusiastic</td>
      <td>1</td>
      <td>14980.950</td>
      <td>0</td>
      <td>9.614535</td>
      <td>0</td>
      <td>1</td>
      <td>0.532268</td>
    </tr>
    <tr>
      <th>2</th>
      <td>39</td>
      <td>enthusiastic</td>
      <td>0</td>
      <td>12342.542</td>
      <td>1</td>
      <td>9.420807</td>
      <td>0</td>
      <td>1</td>
      <td>0.479374</td>
    </tr>
    <tr>
      <th>3</th>
      <td>29</td>
      <td>enthusiastic</td>
      <td>0</td>
      <td>13142.107</td>
      <td>1</td>
      <td>9.483577</td>
      <td>0</td>
      <td>1</td>
      <td>0.466185</td>
    </tr>
    <tr>
      <th>4</th>
      <td>34</td>
      <td>enthusiastic</td>
      <td>1</td>
      <td>13255.874</td>
      <td>0</td>
      <td>9.492196</td>
      <td>0</td>
      <td>1</td>
      <td>0.521715</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_sample2 = data_marathon.sample(n = 2000, random_state = 123)
treatment = data_sample2['vaporfly'].values
outcome = data_sample2['ln_race_time'].values
features = data_sample2[['propensity_score']].values

causal = CausalModel(Y = outcome, D = treatment, X = features)
causal.est_via_matching()
ate2 = causal.estimates['matching']['ate']
ate2 = round(ate2,3)
ate2
```




    -0.042



- **Interpretation**: Using propensity score matching with the variables age, male, seasoned, and enthusiastic, the estimated ATE of wearing Nike Vaporfly shoes on ln_race_time is -0.042. This indicates that, after controlling for these variables, Vaporfly shoes are associated with a reduction in the natural logarithm of race times.

## Part 2: Regression Analysis

#### 1. Regression with Robust Standard Errors


```python
outcome = data_marathon['ln_race_time']
features = data_marathon[['vaporfly','male','seasoned','enthusiastic']]
features = sm.add_constant(features)

model = sm.OLS(outcome, features)
results = model.fit()
ols_robust1 = results.get_robustcov_results()
print(ols_robust1.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           ln_race_time   R-squared:                       0.304
    Model:                            OLS   Adj. R-squared:                  0.303
    Method:                 Least Squares   F-statistic:                     2437.
    Date:                Tue, 15 Oct 2024   Prob (F-statistic):               0.00
    Time:                        23:47:03   Log-Likelihood:                 17850.
    No. Observations:               24699   AIC:                        -3.569e+04
    Df Residuals:                   24694   BIC:                        -3.565e+04
    Df Model:                           4                                         
    Covariance Type:                  HC1                                         
    ================================================================================
                       coef    std err          t      P>|t|      [0.025      0.975]
    --------------------------------------------------------------------------------
    const            9.6532      0.001   7842.522      0.000       9.651       9.656
    vaporfly        -0.0658      0.001    -43.994      0.000      -0.069      -0.063
    male            -0.1285      0.002    -81.963      0.000      -0.132      -0.125
    seasoned        -0.0878      0.002    -39.585      0.000      -0.092      -0.083
    enthusiastic    -0.0547      0.004    -15.446      0.000      -0.062      -0.048
    ==============================================================================
    Omnibus:                      751.285   Durbin-Watson:                   2.006
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              859.354
    Skew:                          -0.402   Prob(JB):                    2.47e-187
    Kurtosis:                       3.434   Cond. No.                         5.72
    ==============================================================================
    
    Notes:
    [1] Standard Errors are heteroscedasticity robust (HC1)
    

- **Interpretation**:

    - **Intercept (const)**: The constant term is 9.6532, indicating the baseline natural log of race time when all predictor variables are zero.
    - **Vaporfly**: The coefficient for Vaporfly is -0.0658, suggesting that wearing Vaporfly shoes decreases the natural log of race time by 0.0658 units on average, holding other factors constant. This result is statistically significant.
    - **Male**: The coefficient for males is -0.1285, indicating that male runners have a lower ln_race_time compared to female runners, holding other factors constant. This result is statistically significant.
    - **Seasoned**: Seasoned marathoners (those with at least 3 prior completed marathons) have a lower ln_race_time by 0.0878 units compared to first-timers, holding other factors constant. This result is statistically significant
    - **Enthusiastic**: Enthusiastic runners (with 1 or 2 prior marathons) have a lower ln_race_time by 0.0547 units compared to first-timers, holding other factors constant. This result is statistically significant.

#### 2. Regression including Age


```python
formula = 'ln_race_time ~ vaporfly + male + seasoned + enthusiastic + age'
ols_model_with_age = smf.ols(formula = formula, data = data_marathon)
ols_robust_with_age = ols_model_with_age.fit().get_robustcov_results()
print(ols_robust_with_age.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           ln_race_time   R-squared:                       0.464
    Model:                            OLS   Adj. R-squared:                  0.464
    Method:                 Least Squares   F-statistic:                     3824.
    Date:                Tue, 15 Oct 2024   Prob (F-statistic):               0.00
    Time:                        23:43:09   Log-Likelihood:                 21081.
    No. Observations:               24699   AIC:                        -4.215e+04
    Df Residuals:                   24693   BIC:                        -4.210e+04
    Df Model:                           5                                         
    Covariance Type:                  HC1                                         
    ================================================================================
                       coef    std err          t      P>|t|      [0.025      0.975]
    --------------------------------------------------------------------------------
    Intercept        9.9345      0.003   2861.720      0.000       9.928       9.941
    vaporfly        -0.0426      0.001    -31.883      0.000      -0.045      -0.040
    male            -0.1296      0.001    -94.077      0.000      -0.132      -0.127
    seasoned        -0.0889      0.002    -45.999      0.000      -0.093      -0.085
    enthusiastic    -0.0548      0.003    -17.544      0.000      -0.061      -0.049
    age             -0.0086      0.000    -84.237      0.000      -0.009      -0.008
    ==============================================================================
    Omnibus:                      570.884   Durbin-Watson:                   1.993
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              646.028
    Skew:                          -0.341   Prob(JB):                    5.21e-141
    Kurtosis:                       3.404   Cond. No.                         186.
    ==============================================================================
    
    Notes:
    [1] Standard Errors are heteroscedasticity robust (HC1)
    

- **Interpretation**:

    - **Intercept (const)**: The constant term is 9.9345, indicating the baseline natural log of race time when all predictor variables are zero
    - **Vaporfly**: The coefficient for Vaporfly is -0.0426, indicating that wearing Vaporfly shoes decreases the natural log of race time by 0.0426 units on average, holding other factors constant. The effect is statistically significant
    - **Male**: The coefficient for males is -0.1296, suggesting that male runners have a lower ln_race_time compared to female runners, holding other factors constant. This is statistically significant
    - **Seasoned**: Coefficient for seasoned unners is -0.0889, indicating better performance compared to first-timers, holding other factors constant. THis is statistically significant
    - **Enthusiastic**: Enthusiastic runners have a coefficient of -0.0548, indicating better performance compared to first-timers, holding other factors constant. This is statistically significant.
    - **Age**: The coefficient for age is -0.0086, indicating that older runners tend to have slightly lower ln_race_time, holding other factors constant. This is statistically significant

 - **Comparison and Conclusion**:

    - The initial model without age shows a Vaporfly coefficient of -0.0658, indicating a stronger initial estimated effect of the Vaporfly shoes
    - When age is included in the model, the Vaporfly coefficient decreases to -0.0426. This change suggests that part of the initial observed effect of Vaporfly shoes was actually due to the age of the runners
    - **R=squared values**: Adding age into the model increases the R-sqaured value from 0.304 to 0.464, indicating that the model with age explains more variability in ln_race_time.

Overall, these results highlight the importance of considering confounding factors, such as age, in the regression model. The adjuisted effect of Vaporfly shoes, after including age, still indicates a statistically significant reduction in race times, thereby supporting the claims that Vaporfly shoes are beneficial for marathon runners.

![Nike Vaporfly](https://github.com/user-attachments/assets/e483b555-a5df-4610-aa55-75c1cbdca91f)
