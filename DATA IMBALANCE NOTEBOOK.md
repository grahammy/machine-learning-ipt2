
# DATA IMBALANCE

Data imbalance this refers to unequal distribution of observation within a dataset mostly on classification task in which some of the observation in dataset may contain large amount of data while other observation contains little of the information.

WHY DATA IMBALANCE?
a)Data imbalance may pose misleading of accurancy.
b)With data imbalance the dataset has underpresented data in which it leads to class distribution skewness. 

DATA TO BE BALANCED
Data to be balanced is the data that will be used as the feature or label on the classification problem.

## Methods For Data Balancing


1)RESAMPLING
A)OVERSAMPLING
This is the process of altering the dataset to remove the imbalancing of data and it is normally conducted on the majority class.The following are some of the mechanism of oversampling:
a)Random oversampling
b)SMOTE
c)ADASYN

B)UNDERSAMPLING
This is the process of throwing away data of the majority class so as to make it match with the minority class.
The following are some of the mechanism of undersampling
a)Random undersampling
b)Near miss
c)Tomeks links
d)Edited nearest neighbors

2)CLASS WEIGHT
This is the process of data imbalance in which weight is provided for each class by mostly considering the the minority class.



```python
#Dealing with imbalance data
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv(r"C:\Users\irwgatap\Documents\ML-week1-challenge-master\data\train_data_week_1_challenge.csv")
data.head()
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
      <th>continue_drop</th>
      <th>student_id</th>
      <th>gender</th>
      <th>caste</th>
      <th>mathematics_marks</th>
      <th>english_marks</th>
      <th>science_marks</th>
      <th>science_teacher</th>
      <th>languages_teacher</th>
      <th>guardian</th>
      <th>internet</th>
      <th>school_id</th>
      <th>total_students</th>
      <th>total_toilets</th>
      <th>establishment_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>continue</td>
      <td>s01746</td>
      <td>M</td>
      <td>BC</td>
      <td>0.666</td>
      <td>0.468</td>
      <td>0.666</td>
      <td>7</td>
      <td>6</td>
      <td>other</td>
      <td>True</td>
      <td>305</td>
      <td>354</td>
      <td>86.0</td>
      <td>1986.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>continue</td>
      <td>s16986</td>
      <td>M</td>
      <td>BC</td>
      <td>0.172</td>
      <td>0.420</td>
      <td>0.172</td>
      <td>8</td>
      <td>10</td>
      <td>mother</td>
      <td>False</td>
      <td>331</td>
      <td>516</td>
      <td>15.0</td>
      <td>1996.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>continue</td>
      <td>s00147</td>
      <td>F</td>
      <td>BC</td>
      <td>0.212</td>
      <td>0.601</td>
      <td>0.212</td>
      <td>1</td>
      <td>4</td>
      <td>mother</td>
      <td>False</td>
      <td>311</td>
      <td>209</td>
      <td>14.0</td>
      <td>1976.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>continue</td>
      <td>s08104</td>
      <td>F</td>
      <td>ST</td>
      <td>0.434</td>
      <td>0.611</td>
      <td>0.434</td>
      <td>2</td>
      <td>5</td>
      <td>father</td>
      <td>True</td>
      <td>364</td>
      <td>147</td>
      <td>28.0</td>
      <td>1911.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>continue</td>
      <td>s11132</td>
      <td>F</td>
      <td>SC</td>
      <td>0.283</td>
      <td>0.478</td>
      <td>0.283</td>
      <td>1</td>
      <td>10</td>
      <td>mother</td>
      <td>True</td>
      <td>394</td>
      <td>122</td>
      <td>15.0</td>
      <td>1889.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data=data.replace({'continue':1, 'drop':0})
```


```python
#showing imbalance data
data.continue_drop.value_counts()
```




    1    16384
    0      806
    Name: continue_drop, dtype: int64




```python
#Using Resampling to solve imbalance data
from sklearn.utils import resample
data_majority=data[data.continue_drop==1]
data_minority=data[data.continue_drop==0]

```


```python
count_1,count_0=data['continue_drop'].value_counts()
```


```python
upsampled=min.sample(counts_1, replace=True)
data_balanced1=pd.concat([upsampled,maj],axis=0)
downsampled=majority.sample(count_1, random_state=123)
data_balanced=pd.concat([upsampled,downsampled],axis=0)

```


```python

```
