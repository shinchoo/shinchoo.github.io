---
layout: single
title:  "Coursera Course Completion Analysis"
subtitle: "Causal Inference 3: Two-Stage Least Squares regression"
categories: python
tag: [Causal Inference, Two-Stage Least Squares regression]
toc: true
---

## Investigating the Impact of Bingeing on Course Completion of Coursera

### Resources
- [Instrumental Variable & Randomized Encouragement Trials: Driving Engagement of Leaners](https://medium.com/coursera-engineering/instrumental-variables-randomized-encouragement-trials-driving-engagement-of-learners-621215e9e3f1)

### Background
Reseachers in Coursera are interested in determing whether certain learning behaviors, such as bingeing, can increase the likelihood of course completion. Bingeing is defined here as completing and starting consecutive weeks of a course on the same day. By using various regression techniques and statistical methods, this project aims to draw conclusions on the impact of bingeing on course completion rates

### Part 1:Linear Regression Analysis

### Data
The dataset contains the following variables for 49,808 learners on Coursera:
- **id**: Unique identifier for each learner
- **paid_enroll**: Dummy variable (1 if learner paid for enrollment, 0 otherwise)
- **prv_wk_nbr**: Most recent course week a learner has completed
- **prv_wk_min**: Minutes a learner spent in the previous week on the platform
- **message**: Dummy variable (1 if learner received an encouraging message, 0 otherwise)
- **binge**: Dummy variable (1 if learner binged, 0 otherwise)
- **complete**: Dummy variable (1 if learner completed the next week in the course, 0 otherwise)


```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from scipy import stats
from linearmodels import IV2SLS
```


```python
data_coursera = pd.read_csv("C:/Users/shincho/OneDrive - KLA Corporation/Desktop/Portfolio/Causal_Inference3/Coursera.csv")
data_coursera.head()
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
      <th>id</th>
      <th>paid_enroll</th>
      <th>prv_wk_nbr</th>
      <th>prv_wk_min</th>
      <th>message</th>
      <th>binge</th>
      <th>complete</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>193</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>194</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>45</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>118</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>247</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### 1. Simple Linear Regression of complete on binge
To begin, we regress the likelihood of course completion (complete) on bingeing behavior (binge) using robust standard errrors


```python
formula = 'complete ~ binge'
model = smf.ols(formula, data = data_coursera).fit()
binge_coeff = round(model.params['binge'],4)
print(binge_coeff)
```

    0.4619
    

- **Result**: The coefficient for binge is 0.4619. This indicates a positive relationship between bingeing and the likelihood of completing the next week in the course. Bingeing increases the likelihood of completing the next week by approximately 46.19%

### 2. Adding Controls: paid_enroll, prv_wk_nbr, prv_wk_min
Next, we included additional control variables to account for other factors that might influence course completion


```python
formula2 = 'complete ~ binge + paid_enroll + prv_wk_nbr + prv_wk_min'
model2 = smf.ols(formula2, data = data_coursera).fit()
binge_coeff2 = round(model2.params['binge'],4)
print(binge_coeff2)
```

    0.3172
    

- **Result**: The coefficient for binge is 0.3172. With additional controls, the coefficient for binge decreases to 31.72%, yet remains significant. This reduction suggests that some of the initial effect observed may be explained by other factors, such as whether a learner paid for the course or how far they've progressed.

### Part 2:Instrumental Variable Analysis
To address potential self-selection bias (learners who binge might also be inherently more motivated to complete the course), we conduct an instrumental Variable (IV) analysis using the randomized encouragement trial. In this trial, learners were randomly assigned to receive a motivational message (message)

![image](https://github.com/user-attachments/assets/adb5282d-1015-44f1-8006-a9506fdcf88c)

### 1. Exclusion Restriction Explanation
In this context, the exclusion restriction means that the treatment (receiving a message) should only affect the outcome (course completion) through the endogenous variable (binge behavior) and not directly


```python
formula3 = 'binge ~ message'
model3 = smf.ols(formula3, data_coursera).fit()
robust_reg3 = model3.get_robustcov_results()
robust_reg3.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>binge</td>      <th>  R-squared:         </th> <td>   0.018</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.018</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   911.9</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 16 Oct 2024</td> <th>  Prob (F-statistic):</th> <td>1.55e-198</td>
</tr>
<tr>
  <th>Time:</th>                 <td>15:30:31</td>     <th>  Log-Likelihood:    </th> <td> -16053.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td> 49808</td>      <th>  AIC:               </th> <td>3.211e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 49806</td>      <th>  BIC:               </th> <td>3.213e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>         <td>HC1</td>       <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>    0.8243</td> <td>    0.002</td> <td>  342.078</td> <td> 0.000</td> <td>    0.820</td> <td>    0.829</td>
</tr>
<tr>
  <th>message</th>   <td>    0.0903</td> <td>    0.003</td> <td>   30.198</td> <td> 0.000</td> <td>    0.084</td> <td>    0.096</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>18923.394</td> <th>  Durbin-Watson:     </th> <td>   2.007</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>52775.181</td>
</tr>
<tr>
  <th>Skew:</th>           <td>-2.131</td>   <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 5.696</td>   <th>  Cond. No.          </th> <td>    2.62</td> 
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors are heteroscedasticity robust (HC1)



The coefficient for message is 0.0903 with a p-value of 0.000. This suggests that receiving a message significantly increases the likelihood of bingeing behavior by about 9.03 percentage points, indicating a strong first stage relationsip.

### 2. Intention-to-Treat (ITT) Effect
We then examine the direct impact of receiving a message on course completion


```python
formula4 = 'complete ~ message'
model4 = smf.ols(formula4, data = data_coursera).fit()
l_change4 = round(model4.params['message'],4)
print(l_change4)
```

    0.0113
    

- **Result**: The ITT effect is 0.0113. Receiving a message increases the likelihood of completing the next week by about 1.13%. Despite appearing to be a small effect, considering the massive scale of online courses, this effect may still be practically significant.
    - No Defiers: "No Defiers" suggests that no learners who received the encouraging message would do the opposite of what the message intends simply by receiving it. In other words, no learners would start the next week's course despite planning not to, just because they received the message

### 3. Share of "Always-Takers"
We calculate the proportion of learners who binge without receiving the message


```python
no_message = data_coursera[data_coursera['message'] == 0]
at_share = round(no_message[no_message['binge'] == 1].shape[0] / no_message.shape[0], 4)
print(at_share)
```

    0.8243
    

- **Result**: The share of always-takers is 0.8243

### 4. Share of "Never-Takers"
We calculate the proportion of learners who did not binge even when they received the message


```python
message = data_coursera[data_coursera['message'] == 1]
nt_share = round(message[message['binge'] == 0].shape[0] / message.shape[0],4)
print(nt_share)
```

    0.0854
    

- **Result**: The share of never-takers is 0.0854

### 5. IV Estimation Calculation
To understand the causal effect of bingeing on completion for compliers (those who binged because of the message), we calculate the IV estimate


```python
iv_estimate = round(l_change4 / robust_reg3.params[1], 4)
print(iv_estimate)
```

    0.1251
    

- **Result**: The IV estimate is 0.1251. Receiving the message leads to an increase in the linkelihood of completing the next week by approximately 12.51% for those who binged as a result of the encouragement

### 6. Two-Stage Least Squares (2SLS) Regression
Finally, we use the Two-Stage Least Squares method from the linearmodels library to precisely estimate our IV model with robust standard errors


```python
data = data_coursera.dropna()

# Two-Stage Least Squares regression
model6 = 'complete ~ 1 + [binge ~ message]'
is2sls6 = IV2SLS.from_formula(model6, data).fit()
print(is2sls6.summary)
```

                              IV-2SLS Estimation Summary                          
    ==============================================================================
    Dep. Variable:               complete   R-squared:                      0.1213
    Estimator:                    IV-2SLS   Adj. R-squared:                 0.1213
    No. Observations:               49808   F-statistic:                    19.395
    Date:                Wed, Oct 16 2024   P-value (F-stat)                0.0000
    Time:                        22:25:55   Distribution:                  chi2(1)
    Cov. Estimator:                robust                                         
                                                                                  
                                 Parameter Estimates                              
    ==============================================================================
                Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
    ------------------------------------------------------------------------------
    Intercept      0.7862     0.0248     31.690     0.0000      0.7376      0.8348
    binge          0.1254     0.0285     4.4040     0.0000      0.0696      0.1812
    ==============================================================================
    
    Endogenous: binge
    Instruments: message
    Robust Covariance (Heteroskedastic)
    Debiased: False
    

The summary indicates that bingeing behavior, when instrumented by the encouragement message, significantly increases the likelihood of course completion. The coefficient for binge is 0.1254, which aligns with our manually calculated IV estimate.

### Conclusion
Our analysis highlights the positive impact of bingeing on course completion for Coursera learners. By leveraging a randomized encouragement trial and statistical methods, we mitigated biases and revealed robust findings. This is valuable for Online lecturers when designing interventions that boost learner engagement and success on Coursera

![image](https://github.com/user-attachments/assets/24a50ee5-c8d8-4e77-968c-dadf94d63dbb)
