---
layout: single
title:  "Labor Market Discrimination"
subtitle: "Causal Inference 1: t-test"
categories: python
tag: [Causal Inference, t-test]
---
## Labor Market Discrimination: Analyzing Racial Discrimination in Hiring Practices

### Resources
- [Bertrand and Mullainathan, 2004](https://www.aeaweb.org/articles?id=10.1257/0002828042002561)

### Background
In this project, we explore potential racial discrimination against African-Americans in the labor market, utilizing two distinct datasets: one observational (survey data) and one experimental. This study is inspired by research conducted by Marianne Bertrand and Sendhil Mullainathan, who sent fictitiouis resumes in response to job adverts in Boston and Chicago to analyze callback rates based on the racial connotation of the names of these resumes

### Part 1: Observational Analysis

### Data
The survey dataset contains information about 10,593 individuals and four key variables:
- **black**: dummy variable for race; equal to 0 if white, 1 if black
- **yearsexp**: years of work experience
- **somecol_more**: 1 if college dropout or college degree or college degree and more, 0 otherwise
- **employed**: 1 is employed, 0 if unemployed


```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.api as sms
from scipy import stats
import warnings
warnings.simplefilter("ignore")

data_survey = pd.read_csv("C:/Users/shincho/OneDrive - KLA Corporation/Desktop/Portfolio/Casual_Inference1/survey.csv")
data_survey.head()
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
      <th>yearsexp</th>
      <th>somecol_more</th>
      <th>col_more</th>
      <th>employed</th>
      <th>black</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### 1. Checking Covariate Balance
We start by examining if the mean of somecol_more differs between African-Americans and Whites


```python
black = data_survey[data_survey['black'] == 1]['somecol_more']
white = data_survey[data_survey['black'] == 0]['somecol_more']

ttest1_1 = stats.ttest_ind(black, white)
print(ttest1_1)
```

    Ttest_indResult(statistic=-11.24095598344323, pvalue=3.7514090837794205e-29)
    

- **Result**: The t-test result shows a static of -11.24 and a p-value of 3.75 x 10<sup>-29</sup>
, indicating a significant difference at the 5% level

### 2. Years of Experience
Next, we test if the mean years of experience (yearsexp) differ between African-Americans and Whites


```python
black_year = data_survey[data_survey['black'] == 1]['yearsexp']
white_year = data_survey[data_survey['black'] == 0]['yearsexp']

ttest1_2 = stats.ttest_ind(black_year, white_year)
print(ttest1_2)
```

    Ttest_indResult(statistic=-20.665913460336267, pvalue=4.743084195169169e-93)
    

- **Result**: The t-test result yields a statistic of -20.67 and a p-value of 4.74 x 10<sup>-93</sup>
, which is highly significant

### 3. Employement Status
Finally, we analyze if the employment status (emplyed) differs between races


```python
black_employed = data_survey[data_survey['black'] == 1]['employed']
white_employed = data_survey[data_survey['black'] == 0]['employed']

ttest_employed = stats.ttest_ind(black_employed, white_employed)
print(ttest_employed)
```

    Ttest_indResult(statistic=-7.1420030594715485, pvalue=9.802071342176246e-13)
    

- **Result**: The t-test result yields a statistic of -7.14 and a p-value of 9.80 x 10<sup>-13</sup>
, indicating a significant difference

### Interpretation
Based on the observational data analysis, we find significant differences in somecol_more, yearsexp, and employed between African-Americans and Whites. However, these differences alone do not establish casual links to racial discrimination due to potentional confounders.

## Part 2: Experimental Analysis

### Data
The experimental dataset includes 4,870 observations with variables like gender, race, education and job experience. The key research is whether resumes with black-sounding names receive fewer callbacks than those with white-sounding names
- **id**: a de-identified identifier to represent unique observations (resumes)
- **male**: dummy variable for gender; equal to 0 if female, 1 if male
- **black**: dummy variable for race; equal to 0 if resume has a white sounding name and 1 if resume has a black sounding name
- **education**: 0 if not reported, 1 if high school dropout, 2 if high school graduate, 3 if college dropout, 4 if has college degree or more
- **computerskills**: 1 if resume mentions some computer skills, 0 otherwise
- **ofjobs**: number of jobs listed on the resume
- **yearsexp**: number of years of work experience on the resume
- **call**: 1 if applicant was called back, 0 otherwise


```python
data_random = pd.read_csv("C:/Users/shincho/OneDrive - KLA Corporation/Desktop/Portfolio/Casual_Inference1/random_exper.csv")
data_random.head()
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
      <th>education</th>
      <th>ofjobs</th>
      <th>yearsexp</th>
      <th>computerskills</th>
      <th>call</th>
      <th>id</th>
      <th>male</th>
      <th>black</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>2</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>3</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>4</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>3</td>
      <td>22</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### 1. Treatment and Outcome Variable


```python
d_i = data_random['black']
y_i = data_random['call']
```

### 2. Treatment Status and Counterfactual Outcome
Checking the treatment status for individual with id = 345


```python
individual = data_random.loc[data_random['id'] == 345]
individual
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
      <th>education</th>
      <th>ofjobs</th>
      <th>yearsexp</th>
      <th>computerskills</th>
      <th>call</th>
      <th>id</th>
      <th>male</th>
      <th>black</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>344</th>
      <td>4</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>345</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



- **Result**: The individual has a black-sounding name (black=1), making the observed outcome Y_b_345. The counterfactual outcome would be Y_w_345

### 3. Covariate Balance
We check if other variables like gender, computer skills, education, and job experience are balanced across races.


```python
male_black = pd.crosstab(data_random['male'],data_random['black'])
computerskills_black = pd.crosstab(data_random['computerskills'],data_random['black'])
education_black = pd.crosstab(data_random['education'],data_random['black'])
ofjobs_black = pd.crosstab(data_random['ofjobs'],data_random['black'])

print(male_black)
print(computerskills_black)
print(education_black)
print(ofjobs_black)
```

    black     0     1
    male             
    0      1860  1886
    1       575   549
    black              0     1
    computerskills            
    0                466   408
    1               1969  2027
    black         0     1
    education            
    0            18    28
    1            18    22
    2           142   132
    3           513   493
    4          1744  1760
    black     0    1
    ofjobs          
    1        54   56
    2       347  357
    3       726  703
    4       800  811
    5       258  275
    6       243  221
    7         7   12
    

The variables appear balanced across races, confirming the experiment's randomization effectiveness

### 4. Callback Rates
Testing if callback rates differ between black-sounding and white-sounding names:


```python
black_call = data_random[data_random['black'] == 1]['call']
white_call = data_random[data_random['black'] == 0]['call']

ttest2_1 = stats.ttest_ind(black_call, white_call)
print(ttest2_1)
```

    Ttest_indResult(statistic=-4.114705266723095, pvalue=3.9408025140695284e-05)
    

- **Result**: The t-test result yields a statistic of -4.11 and a p-value of 3.94 x 10<sup>-5</sup>
, indicating a significant difference

### 5. Gender-Specific Analysis
#### Females


```python
female_black_call = data_random[(data_random['black'] == 1) & (data_random['male'] == 0)]['call']
female_white_call = data_random[(data_random['black'] == 0) & (data_random['male'] == 0)]['call']

ttest2_2 = stats.ttest_ind(female_black_call, female_white_call)
print(ttest2_2)
```

    Ttest_indResult(statistic=-3.6369213964305627, pvalue=0.0002796319942029361)
    

- **Result**: Significant difference with a p-value of 0.00028

#### Males


```python
male_black_call = data_random[(data_random['black'] == 1) & (data_random['male'] == 1)]['call']
male_white_call = data_random[(data_random['black'] == 0) & (data_random['male'] == 1)]['call']

ttest2_3 = stats.ttest_ind(male_black_call, male_white_call)
print(ttest2_3)
```

    Ttest_indResult(statistic=-1.9501711134984252, pvalue=0.05140448724722174)
    

- **Result**: Significant difference with a p-value of 0.051

### Conclusion
The experimental data provide strong evidence of racial discrimination in callback rates for job applications with black-sounding names. This contrasts with the observational data, where causation could not be established. Thus, randomized experiments like the one conducted by Bertrand and Mullainathan are crucial for identifying discriminatory practices in the labor market.
