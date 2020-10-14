# **Fusion of algorithms for face recognition**

###Realised By : Khalil Bouguerra 
---


####For a full version of the report and  better graphics visulisation please open this notebook in [google colab environment [Fusion of algorithms for face recognition](https://colab.research.google.com/drive/1obJMHbaiCY82qiV152AzK2OQ4V3ee3ah?usp=sharing)
---



# Introduction 


The increasingly ubiquitous presence of biometric solutions and face recognition in particular in everyday life requires their adaptation for practical scenario. In the presence of several possible solutions, and if global decisions are to be made, each such single solution can be far less efficient than tailoring them to the complexity of an image.

In this challenge, the goal is to build a fusion of algorithms in order to construct the best suited solution for comparison of a pair of images. This fusion will be driven by qualities computed on each image.

Comparing of two images is done in two steps.

*  1st, a vector of features is computed for each image.
*  2nd, a simple function produces a vector of scores for a pair of images. 

$\implies$ The goal is to create a function that will compare a pair of images based on the information mentioned above, and decide whether two images belong to the same person.

# Work OverFlow : 

## Imports 


```python
import pandas as pd 
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
import seaborn as sns
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pprint import pprint as Print
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.ensemble import BaggingClassifier
import  lightgbm as lgb
import time
from sklearn.model_selection import StratifiedShuffleSplit
import pickle
from scipy.stats import randint 
from scipy.stats import uniform 
from google.colab import files
import plotly.graph_objects as go
import plotly.express as px
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import RFECV, SelectKBest, f_classif

verbose=False 
max_coef=3
CM=False 

Bare_model_scores={}
tuned_model_scores={}
Meta_model_scores={}

```

##Importing Data 



```python
from google_drive_downloader import GoogleDriveDownloader as gdd
gdd.download_file_from_google_drive(file_id='1K30Xd4z_hF77f7htkHZwIwkqohTwpfli',dest_path='./data/xtrain_challenge.csv')
gdd.download_file_from_google_drive(file_id='1DbpcvlOxDMk64acnQZcJLAmGm-Gu7M8e',dest_path='./data/ytrain_challenge.csv')
gdd.download_file_from_google_drive(file_id='17claJ2kZxXBJ8NjydXTJtdYPiYbmO7Tt',dest_path='./data/xtest_challenge.csv')
df = pd.read_csv('./data/xtrain_challenge.csv')
df['y']= pd.read_csv('./data/ytrain_challenge.csv')['y']
Test=pd.read_csv('./data/xtest_challenge.csv')

```

    Downloading 1K30Xd4z_hF77f7htkHZwIwkqohTwpfli into ./data/xtrain_challenge.csv... 

##Data Spliting & Scalling 

### Splitting the data: 

* Training data:  This is the data used to train the model, to fit the model parameters. It will account for the largest proportion of data as we wish for the model to see as many examples as possible.

* Test data :This is the data used to evaluate and compare models. As this data has not been seen during training nor tuning, it can provide insight into whether your models generalize well to unseen data.

### Scalling

* Scalling or Standardization  of the data 

* Since the range of values of raw data varies widely, in some machine learning algorithms, objective functions will not work properly without normalization. 
In our case , many classifiers calculate the distance between two points by the Euclidean distance. If one of the features has a broad range of values, the distance will be governed by this particular feature. Therefore, the range of all features should be normalized so that each feature contributes approximately proportionately to the final distance.

* Another reason why feature scaling is applied is that gradient descent converges much faster with feature scaling than without it.


### Whole Data Set


```python
X_base=df.drop('y',axis=1)
y_base=df['y']
#spliting the whole dataset:  into 67% for training and 33% for testing( validation)
X_base_train , X_base_test, y_base_train, y_base_test= train_test_split(X_base, y_base, test_size=0.33,random_state =666)
#Scaling the whole dataset and the Testing dataset: 
scaler=StandardScaler() 
scaler.fit(X_base_train)
X_base_train=pd.DataFrame(scaler.transform(X_base_train),columns=X_base_train.columns)
X_base_test=pd.DataFrame(scaler.transform(X_base_test),columns=X_base_test.columns)
Test_scaled=pd.DataFrame(scaler.transform(Test),columns=Test.columns)
```

### SubSample from data set 

$\implies$We need this subsample of the data to tune our models in order to determine the right values for the hyperparamters.
Knowing that tunning the model consisted in fitting the models multiple times using different combinations of hyperparamters values. 
This step is computationally consuming, especially when the train dataset is this large. So we decided to use a sub-sample of the training data to use it as a representative sample of our data during the tuning phase in order to make the computations less time consuming (around 10hours for some algorithms running just on 10% of the data)


```python
#We will run our models tunning on 10% training 10% testing (for computational reasons) 
y=np.array(y_base)
sss = StratifiedShuffleSplit( n_splits=1, test_size=0.8, random_state=666)

## The diffrence between StratifiedShuffleSplit and train_test_split 
## is that train_test_split splits the data randomly without taking in consideration
## the proprtion of each class in each item of the split.
## We can make sure the classes proportions is split evenly if we use StratifiedShuffleSplit

for split_1,split_2 in sss.split(np.zeros(len(y)),y):
  X_tunning=X_base.iloc[split_1] #20% of  X_train  
  y_tunning=y_base[split_1] #20% of y_train

## even for tunning we need train (tunning) set and test set for tunned models selection
## that is why we split the tunning data into  train and test sets evenly 
## which result in 10% of the whole training data for each of tunnig train and test sets 

X_tunning_train,X_tunning_test,y_tunning_train,y_tunning_test=train_test_split(X_tunning,y_tunning,test_size=0.5,random_state=666) #10% of data for tunning_train & 10% of data for  tunning_test 

#Scalling 
scaler=StandardScaler() 
scaler.fit(X_tunning_train)
X_tunning_train=pd.DataFrame(scaler.transform(X_tunning_train),columns=X_tunning_train.columns)
X_tunning_test=pd.DataFrame(scaler.transform(X_tunning_test),columns=X_tunning_test.columns)
  
```

##Exploratary Data Analysis


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1068504 entries, 0 to 1068503
    Data columns (total 38 columns):
     #   Column  Non-Null Count    Dtype  
    ---  ------  --------------    -----  
     0   qs1     1068504 non-null  float64
     1   qs2     1068504 non-null  float64
     2   qs3     1068504 non-null  float64
     3   qs4     1068504 non-null  float64
     4   qs5     1068504 non-null  float64
     5   qs6     1068504 non-null  float64
     6   qs7     1068504 non-null  float64
     7   qs8     1068504 non-null  float64
     8   qs9     1068504 non-null  float64
     9   qs10    1068504 non-null  float64
     10  qs11    1068504 non-null  float64
     11  qs12    1068504 non-null  float64
     12  qs13    1068504 non-null  float64
     13  qr1     1068504 non-null  float64
     14  qr2     1068504 non-null  float64
     15  qr3     1068504 non-null  float64
     16  qr4     1068504 non-null  float64
     17  qr5     1068504 non-null  float64
     18  qr6     1068504 non-null  float64
     19  qr7     1068504 non-null  float64
     20  qr8     1068504 non-null  float64
     21  qr9     1068504 non-null  float64
     22  qr10    1068504 non-null  float64
     23  qr11    1068504 non-null  float64
     24  qr12    1068504 non-null  float64
     25  qr13    1068504 non-null  float64
     26  s1      1068504 non-null  float64
     27  s2      1068504 non-null  float64
     28  s3      1068504 non-null  float64
     29  s4      1068504 non-null  float64
     30  s5      1068504 non-null  float64
     31  s6      1068504 non-null  float64
     32  s7      1068504 non-null  float64
     33  s8      1068504 non-null  float64
     34  s9      1068504 non-null  float64
     35  s10     1068504 non-null  float64
     36  s11     1068504 non-null  float64
     37  y       1068504 non-null  int64  
    dtypes: float64(37), int64(1)
    memory usage: 309.8 MB


###NA Values
In real life dataset we usually end up with some missing values (aka NA, None, NaN,.. values)

Some algorithms can't deal with such data type (NA) or it  affects its performances. So we should locate these values and either drop the row, which may cause  a lot of information loss or change it to some other value depending on the case (null value, mean value, .. )


```python
df.isna().sum(axis=0)
## We don't have to worry  about NA values!
```




    qs1     0
    qs2     0
    qs3     0
    qs4     0
    qs5     0
    qs6     0
    qs7     0
    qs8     0
    qs9     0
    qs10    0
    qs11    0
    qs12    0
    qs13    0
    qr1     0
    qr2     0
    qr3     0
    qr4     0
    qr5     0
    qr6     0
    qr7     0
    qr8     0
    qr9     0
    qr10    0
    qr11    0
    qr12    0
    qr13    0
    s1      0
    s2      0
    s3      0
    s4      0
    s5      0
    s6      0
    s7      0
    s8      0
    s9      0
    s10     0
    s11     0
    y       0
    dtype: int64



###Duplicated data
* Duplicated values if present with high occurence in our dataset can affect the performance of the models. 
* So we better check for duplicated values and eliminate them if found.


```python
df[df.duplicated(keep="first")].count() 
## We don't have to worry about duplicated rows in our dataset. 
```




    qs1     0
    qs2     0
    qs3     0
    qs4     0
    qs5     0
    qs6     0
    qs7     0
    qs8     0
    qs9     0
    qs10    0
    qs11    0
    qs12    0
    qs13    0
    qr1     0
    qr2     0
    qr3     0
    qr4     0
    qr5     0
    qr6     0
    qr7     0
    qr8     0
    qr9     0
    qr10    0
    qr11    0
    qr12    0
    qr13    0
    s1      0
    s2      0
    s3      0
    s4      0
    s5      0
    s6      0
    s7      0
    s8      0
    s9      0
    s10     0
    s11     0
    y       0
    dtype: int64



### Outliers:

####Outliers Elmination


```python
##Original Data 
img1=df[df.columns[:13]].drop(['qs10'],axis=1) ##we didnt use qs10 because with such large values we can't see outlires in boxplot 
img2=df[df.columns[13:26]].drop(['qr10'],axis=1)  ##we didnt use qr10 because with such large values we can't see outlires in  boxplot 
score=df[df.columns[26:-1]]

## Data after Eliminating outliers: 

df_2= df[(np.abs(stats.zscore(df)) < 6).all(axis=1)] 

## Removing outlires(we consider values  very far from the mean as outliers aka having a large Zscore )
## This just trims any observations with values four standard deviations from the mean or more (either positive or negative since we are taking the absolute value).
## => we should check if this makes the results  any better later on.

img1_2=df_2[df_2.columns[:13]].drop(['qs10'],axis=1)  
img2_2=df_2[df_2.columns[13:26]].drop(['qr10'],axis=1) 
score_2=df_2[df_2.columns[26:-1]]
print("Number of observations to be dropped if we use outliers elimination: ", df.count()['y']-df_2.count()['y']) 
print("Number of observations in positive class before dropping outliers:", df[df['y']==1].count()['y']/df['y'].count())
print("Number of observations in positive class after  dropping outliers:",df_2[df_2['y']==1].count()['y']/df_2['y'].count())

#First, I want to go ahead and check for outliers. I’ll use a Seaborn boxplot for this,

sns.set(rc={'figure.figsize':(18,54)})
fig, axs = plt.subplots(ncols=2,nrows=3)
sns.boxplot(data=img1,ax=axs[0][0])
sns.boxplot(data=img1_2,ax=axs[0][1])
sns.boxplot(data=img2,ax=axs[1][0])
sns.boxplot(data=img2_2,ax=axs[1][1])
sns.boxplot(data=score,ax=axs[2][0])
sns.boxplot(data=score_2,ax=axs[2][1])

```

    Number of observations to be dropped if we use outliers elimination:  30894
    Number of observations in positive class before dropping outliers: 0.03705461093266848
    Number of observations in positive class after  dropping outliers: 0.02568402386253024





    <matplotlib.axes._subplots.AxesSubplot at 0x7f73e2fef320>




![png](output_23_2.png)


####Outliers effect  using a simple model as a baseline : 


```python
def logReg(X_train , y_train, X_test , y_test, **kwargs): 
  logreg=LogisticRegression(**kwargs)
  logreg.fit(X_train, y_train)
  ranked_coef=list(zip(X_train.columns,logreg.coef_[0]))
  ranked_coef.sort(key=lambda x:-x[1])
  Print(ranked_coef[:3])
  y_pred=logreg.predict(X_test)
  print("="*20)
  print("Classification Accuracy: ",accuracy_score(y_test,y_pred))
  print("F1 score: ",f1_score(y_test,y_pred))
  print("="*20)
  print("Confusion Matrix:")
  print(confusion_matrix(y_test, y_pred))
  print("="*20)
  return logreg



```


```python
##Testing the performance of a baseline model after droping outliers 

X_outlier=df_2.drop('y',axis=1)
y_outlier=df_2['y']
X_outlier_train , X_outlier_test, y_outlier_train, y_outlier_test= train_test_split(X_outlier, y_outlier, test_size=0.33,random_state =666)

scaler=StandardScaler() 
scaler.fit(X_outlier_train)
X_outlier_train_scaled=pd.DataFrame(scaler.transform(X_outlier_train),columns=X_outlier_train.columns)
X_outlier_test_scaled=pd.DataFrame(scaler.transform(X_outlier_test),columns=X_outlier_test.columns)
x=logReg(X_outlier_train_scaled, y_outlier_train,X_outlier_test_scaled,y_outlier_test,max_iter=2000)

```

    [('s5', 1.6360795781397088),
     ('s4', 1.1514490523914094),
     ('s9', 1.1340366947616451)]
    ====================
    Classification Accuracy:  0.9982594067964907
    F1 score:  0.9654732939404471
    ====================
    Confusion Matrix:
    [[333483    206]
     [   390   8333]]
    ====================



```python
##Testing the performance of a baseline model after droping outliers 
X_base=df.drop('y',axis=1)
y_base=df['y']
X_base_train , X_base_test, y_base_train, y_base_test= train_test_split(X_base, y_base, test_size=0.33,random_state =666)
scaler=StandardScaler() 
scaler.fit(X_base_train)
X_base_train_scaled=pd.DataFrame(scaler.transform(X_base_train),columns=X_base_train.columns)
X_base_test_scaled=pd.DataFrame(scaler.transform(X_base_test),columns=X_base_test.columns)
Test_scaled=pd.DataFrame(scaler.transform(Test),columns=Test.columns)
x=logreg=logReg(X_base_train_scaled, y_base_train,X_base_test_scaled,y_base_test,max_iter=2000)

```

    [('s5', 2.200376840079408),
     ('s9', 1.601991146321581),
     ('s4', 1.502850492514285)]
    ====================
    Classification Accuracy:  0.9981338997807757
    F1 score:  0.9744068455853753
    ====================
    Confusion Matrix:
    [[339423    236]
     [   422  12526]]
    ====================


$\implies$ As you can see in the boxplots we managed to eliminate outliers from our data, but it turns out that what we qualified as outliers are actually the positive class observation and we can see that after we eliminate what we previously considered as outliers we lost most of the positive class observations (using Z score threshold = 6) and all of them (using score threshold= 8 )

$\implies$ Using Zscore threshold = 6 as a criteria to remove outliers actually improved  our baseline model performance (slight increase in accuracy and F1 score plus  the confusion matrix proves that we have less wrong classification ). 

 $\implies$ This may seems as a good thing since we improved the performance , But actually NO , because the improvement is due to the fact that we have less data in the postive class  ( one way to see it is that we removed the obeservation that are hard to classify , because they are far from the mean data => Thus improving the metrics and not the  performance of the model ) 

$\implies$ Thus we won't be eliminating those observations , since it will make our data useless without a positive class.

####Corrolation Analysis 


Let's check corolation  between our target feature and  our independent variables. 



```python
def heatmap(df): 
  corr=df.corr()
  fig,ax=plt.subplots(figsize=(30,30))
  mask=np.zeros_like(corr,dtype=np.bool)
  color_map=sns.color_palette("hot_r")
  ax=sns.heatmap(corr,cmap=color_map,mask=mask,annot=True)
  most_corr_features=corr['y'].drop('y')
  return most_corr_features.nlargest(10)
```


```python
most_corr_features= heatmap(df)
print(most_corr_features)
```

    s9     0.846801
    s4     0.839741
    s11    0.838700
    s10    0.836195
    s5     0.832402
    s7     0.824881
    s8     0.809865
    s6     0.790947
    s3     0.786453
    s1     0.764015
    Name: y, dtype: float64



![png](output_31_1.png)


$\implies$ It looks like our target variable has a mostly weak correlation, with the exception of the scores features  that are  the most corrolated features with the target varible,
and this explains why with a simple LogReg we found the most inmportant features are {s5,s9,s4} , because they are the most corrolated ones with the target variable.


###Unbalanced/Balanced Data:  


Imbalanced classes are a common problem in machine learning classification where there are a disproportionate ratio of observations in each class. Let's check if we have this problem of class imbalance.
 



```python

class_0_count=df[df['y']==0].count()['y']
class_1_count=df[df['y']==1].count()['y']
all_data_count=df.count()['y']
class_1_ratio= class_1_count/all_data_count
class_0_ratio= class_0_count/all_data_count
print(class_1_ratio,class_0_ratio)
fig = go.Figure()
fig.add_trace(go.Pie( values=[class_1_ratio,class_0_ratio], labels=["Postive Class","Negative Class"]))
fig.update_layout(title="Proportions of Classes among the whole  Data",

    font=dict(
        family="Courier New, monospace",
        size=14,
        color="#7f7f7f"
    ))
fig.show()

```

    0.03705461093266848 0.9629453890673315



<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="e05eb016-97d5-4be4-980c-f35e65ac1368" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("e05eb016-97d5-4be4-980c-f35e65ac1368")) {
                    Plotly.newPlot(
                        'e05eb016-97d5-4be4-980c-f35e65ac1368',
                        [{"labels": ["Postive Class", "Negative Class"], "type": "pie", "values": [0.03705461093266848, 0.9629453890673315]}],
                        {"font": {"color": "#7f7f7f", "family": "Courier New, monospace", "size": 14}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Proportions of Classes among the whole  Data"}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('e05eb016-97d5-4be4-980c-f35e65ac1368');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };

            </script>
        </div>
</body>
</html>


$\implies$ We have  a disproportionate ratio of observations in each class (3.7% vs 96.3%).

Most machine learning algorithms work best when the number of samples in each class are about equal. This is because most algorithms are designed to maximize accuracy and reduce error. 

So this kind of data distrubution can cause some problems later on like : 

  * The model predicting always The major class 

  * The Performance won't be as expected because we don't have enough data in Positive class.
 

Solutions: 

  * Resampling Techniques — Oversample minority class:
  
  Oversampling can be defined as adding more copies of the minority class.
    * Pros: Oversampling can be a good choice when you don’t have a ton of data to work with.
    * Cons: The computation Cost increases with the data increasing.
  * Resampling techniques — Undersample majority class: 
  Undersampling can be defined as removing some observations of the majority class. 
    * Pros: Undersampling can be a good choice when we have a ton of data.
    * Cons: Removing information that may be valuable  could lead to underfitting and poor generalization to the test set.

$\implies$ we will use oversimpling the data  in order to compensate for the data unbalance without losing what could be valuable information.
  



```python

# concatenate our training data back together
X = pd.concat([X_base_train, y_base_train], axis=1)

# separate minority and majority classes
Pos_class = X[X['y']==1]
Neg_class = X[X['y']==0]

ratio=4
# upsample minority
Pos_upsampled = resample(Pos_class,
                          replace=True, # sample with replacement
                          n_samples=ratio*len(Pos_class), # we can't resample the data alot (like len(Neg_class))
                                                      # because the data will be redendent alot and that may 
                                                      # effect out results later on. Exprementing some values for ratio 
                                                      # we found out that 4 gives the good results. 
                          random_state=666) 

# combine majority and upsampled minority
upsampled = pd.concat([Neg_class,Pos_upsampled])

X_base_train=upsampled.drop('y',axis=1)
y_base_train = upsampled['y']

print(upsampled[upsampled['y']==1].count()['y'])



class_0_count=upsampled[upsampled['y']==0].count()['y']
class_1_count=upsampled[upsampled['y']==1].count()['y']
all_data_count=upsampled.count()['y']
class_1_ratio= class_1_count/all_data_count
class_0_ratio= class_0_count/all_data_count
print(class_1_ratio,class_0_ratio)
fig = go.Figure()
fig.add_trace(go.Pie( values=[class_1_ratio,class_0_ratio], labels=["Postive Class","Negative Class"]))
fig.update_layout(title="Proportions of Classes among the Train Data",
    font=dict(
        family="Courier New, monospace",
        size=14,
        color="#7f7f7f"
    ))
fig.show()

```

    106580
    0.13392273746217795 0.866077262537822



<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="fd2cf5ab-9efd-48b5-a37f-a9968cc369fc" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("fd2cf5ab-9efd-48b5-a37f-a9968cc369fc")) {
                    Plotly.newPlot(
                        'fd2cf5ab-9efd-48b5-a37f-a9968cc369fc',
                        [{"labels": ["Postive Class", "Negative Class"], "type": "pie", "values": [0.13392273746217795, 0.866077262537822]}],
                        {"font": {"color": "#7f7f7f", "family": "Courier New, monospace", "size": 14}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Proportions of Classes among the Train Data"}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('fd2cf5ab-9efd-48b5-a37f-a9968cc369fc');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };

            </script>
        </div>
</body>
</html>


##Implementation

###Model serialization 


```python
#SAVING Models,Variables,parameters  on hard disk 
def save_model(model,model_name,acc): 
  path=PATH+model_name+"-"+str(acc)+".mod"
  pickle.dump(model, open(path, 'wb')) 
def save_param(param,model_name,acc): 
  path=PATH+model_name+"-"+str(acc)+".prm"
  pickle.dump(param, open(path, 'wb')) 
def save_feat(feat,model_name): 
  path=PATH+model_name+".ft"
  pickle.dump(feat, open(path, 'wb'))
#LOADING Models,Variables,parameters on hard sisk 
def load(name): 
  return pickle.load(open(PATH+name, 'rb'))
```

### Metrics 


```python
def scores(y_test,y_pred):
  acc=accuracy_score(y_test,y_pred)
  precision=precision_score(y_test,y_pred)
  recall=recall_score(y_test,y_pred)
  f1=f1_score(y_test,y_pred)
  if(verbose):
    print("="*40)
    print("Classification Accuraccy: ",format(acc,'.10f'))
    print("precision score : ",format(precision,'.10f'))
    print("recall score: ",format(recall,'.10f'))
    print("f1 score: ",format(f1,'.10f'))
    print("="*40)
  return [acc,precision,recall,f1]

```

### Baseline Models: 

#### LogisticRegression 



```python
def logReg(X_train, y_train, X_test, y_test,**kwargs):
    logreg = LogisticRegression(max_iter=2000,**kwargs)
    logreg.fit(X_train, y_train)
    if(verbose): 
      ranked_coef=list(zip(X_train.columns,logreg.coef_[0]))
      ranked_coef.sort(key=lambda x:-x[1])
      Print(ranked_coef[:max_coef])  
    y_pred = logreg.predict(X_test)
    Bare_model_scores['logreg']=scores(y_test, y_pred)
    if(CM): 
      plot_confusion_matrix(logreg,X_train,y_train,cmap='Blues')
    return logreg
```

####RidgeClassifier 


```python
def ridge(X_train, y_train, X_test, y_test,**kwargs):
    ridgeclassifier = RidgeClassifier(**kwargs)
    ridgeclassifier.fit(X_train, y_train)
    if(verbose): 
      ranked_coef=list(zip(X_train.columns,ridgeclassifier.coef_[0]))
      ranked_coef.sort(key=lambda x:-x[1])
      Print(ranked_coef[:max_coef]) 
    y_pred = ridgeclassifier.predict(X_test)
    Bare_model_scores['ridge']=scores(y_test, y_pred)
    if(CM): 
      plot_confusion_matrix(ridgeclassifier,X_train,y_train,cmap='Blues')
    return ridgeclassifier
```

####SVM


```python
def svm(X_train, y_train, X_test, y_test, **kwargs):
    svm = SVC(**kwargs)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    Bare_model_scores['svm']=scores(y_test, y_pred)
    if(CM):
      plot_confusion_matrix(svm,X_train,y_train,cmap='Blues')
    return svm
```

####K-Nearest Neighbors


```python
def knn(X_train, y_train, X_test, y_test,**kwargs):
    Knn = KNeighborsClassifier(**kwargs)
    Knn.fit(X_train, y_train)
    y_pred = Knn.predict(X_test)
    Bare_model_scores['knn']=scores(y_test, y_pred)
    if(CM): 
      plot_confusion_matrix(Knn,X_train,y_train,cmap='Blues')
    return Knn
```

####BaggingClassifier


```python
def bagging(X_train, y_train, X_test, y_test,**kwargs):
    Bagging = BaggingClassifier( **kwargs) 
    Bagging.fit(X_train,y_train)
    y_pred = Bagging.predict(X_test)
    Bare_model_scores['bagging']=scores(y_test, y_pred)
    if(CM):
      plot_confusion_matrix(Bagging,X_train,y_train,cmap='Blues')
    return Bagging
```

####RandomForest



```python
def randomForest(X_train, y_train, X_test, y_test,**kwargs):
    rf = RandomForestClassifier(**kwargs) 
    rf.fit(X_train,y_train)
    y_pred = rf.predict(X_test)
    Bare_model_scores['randomForest']=scores(y_test, y_pred)
    if(CM): 
      plot_confusion_matrix(rf,X_train,y_train,cmap='Blues')
    return rf
```

####XGBoost


```python
def xgboost(X_train, y_train, X_test, y_test,**kwargs):
    xg = xgb.XGBClassifier(**kwargs)
    xg.fit(X_train,y_train)
    y_pred = xg.predict(X_test)
    Bare_model_scores['xgboost']=scores(y_test, y_pred)
    if(CM): 
      plot_confusion_matrix(xg,X_train,y_train,cmap='Blues')
    return xg
```

####LightGBM


```python
def lgbm(X_train, y_train, X_test, y_test,**kwargs):
    lgbm = lgb.LGBMClassifier(**kwargs)
    lgbm.fit(X_train,y_train)
    y_pred = np.rint(lgbm.predict(X_test))
    Bare_model_scores['lgbm']=scores(y_test, y_pred)
    if(CM): 
      plot_confusion_matrix(lgbm,X_train,y_train,cmap='Blues')
    return lgbm


```

#### Performances of the baseline models: 


```python
Models={"logreg":logReg,"ridge":ridge,"svm":svm,"knn":knn,"bagging":bagging,"randomForest":randomForest,"xgboost":xgboost,"lgbm":lgbm}

start_time = time.time()

for model in Models: 

  trained_model=Models[model](X_base_train, y_base_train,X_base_test,y_base_test)
  save_model(trained_model,model,Bare_model_scores[model][0])
  print("---Training :  %s s ---" % (time.time() - start_time))
  start_time=time.time()
  
```

    
    ===== Model:logreg ========
    ---Training :  293.8022964000702 s ---
    ===== Model:ridge ========
    ---Training :  1.223541021347046 s ---
    ===== Model:svm ========
    ---Training :  462.3825237751007 s ---
    ===== Model:knn ========
    ---Training :  1572.4250235557556 s ---
    ===== Model:bagging ========
    ---Training :  680.9955291748047 s ---
    
    
    ===== Model:randomForest ========
    ---Training :  1043.4162375926971 s ---
    ===== Model:xgboost ========
    ---Training :  141.03545260429382 s ---
    ===== Model:lgbm ========
    ---Training :  19.371358633041382 s ---



```python
Models=["lgbm","logreg","ridge","svm","knn","bagging","randomForest","xgboost"]

models=[i for i in Bare_model_scores ]
acc=[Bare_model_scores[i][0] for i in Bare_model_scores ]
f1=[Bare_model_scores[i][-1]for i in Bare_model_scores ]

fig = go.Figure()
fig.add_trace(go.Bar(
    x=models,
    y=acc,
    name='Accuracy',
    marker_color='indianred'
))
fig.add_trace(go.Bar(
    x=models,
    y=f1,
    name='F1',
    marker_color='lightsalmon'
))

fig.update_layout(barmode='group', yaxis=dict(range=[0.9, 1 ]),xaxis_tickangle=-45,title="Accuracy & F1 Score ",
    xaxis_title="Algorithms",
    yaxis_title="Scores",
    font=dict(
        family="Courier New, monospace",
        size=14,
        color="#7f7f7f"
    ))
fig.show()

```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="b1facd4b-bb1b-41e1-a329-2e8d83e427ea" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("b1facd4b-bb1b-41e1-a329-2e8d83e427ea")) {
                    Plotly.newPlot(
                        'b1facd4b-bb1b-41e1-a329-2e8d83e427ea',
                        [{"marker": {"color": "indianred"}, "name": "Accuracy", "type": "bar", "x": ["logreg", "ridge", "svm", "knn", "rf", "xgb", "lgbm", "bagging"], "y": [0.9797506390958892, 0.9634958723761228, 0.9812217256644158, 0.9869489906753376, 0.9890589906753376, 0.9918149881786708, 0.995020769061057, 0.951164249236769]}, {"marker": {"color": "lightsalmon"}, "name": "F1", "type": "bar", "x": ["logreg", "ridge", "svm", "knn", "rf", "xgb", "lgbm", "bagging"], "y": [0.9597506390958892, 0.8634958723761228, 0.9512217256644158, 0.9569489906753376, 0.9770589906753376, 0.9882498817867074, 0.9889020769061057, 0.881164249236769]}],
                        {"barmode": "group", "font": {"color": "#7f7f7f", "family": "Courier New, monospace", "size": 14}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Accuracy & F1 Score "}, "xaxis": {"tickangle": -45, "title": {"text": "Algorithms"}}, "yaxis": {"range": [0.9, 1], "title": {"text": "Scores"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('b1facd4b-bb1b-41e1-a329-2e8d83e427ea');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };

            </script>
        </div>
</body>
</html>


$\implies$  Based on the graph above , the best base models are LightGBM and XGBoost followed by  randomForest. That is why in the rest of this notebook we will focus on those models  

### Features Selection: 

* Feature Selection hugely impacts the performance of model, espacialy if  not all th data is relevent. Irrelevant or partially relevant features can negatively impact model performance.
* Feature Selection is the process where we automatically or manually select those features which contribute most to our prediction variable or output in which we are interested in.


##### Correlation: 

* Correlation states how the features are related to each other or the target variable.
* Using the heatmap plotted above, we can say that the most relevant features are the score features. 
* After some experimenting we figured that, although the output variable is mostly correlated with score features, trying baseline models over those features only, did in fact reduce computation time, but yet decreased the model's performance. 
 
 $\implies$ We will use Feature ranking with recursive feature elimination and cross-validated selection function implemented in most tree based algorithms of sklearn or those algorithms who provide a feature importance score in order to do feature selection and reduce our data space

##### Feature ranking with recursive feature elimination and cross-validated selection: 

* *Recursive* feature elimination is based on the idea to repeatedly construct a model  and choose the worst performing feature ( based on features importance coefficients), setting the feature aside and then repeating the process with the rest of the features. This process is applied until all features in the dataset are exhausted. Features are then ranked according to when they were eliminated. As such, finding the best performing subset of features.


```python
def rfe_cv(estimator,step,cv,scoring,X_train,y_train):
    selector = RFECV(estimator=estimator, step=step, cv=cv, scoring=scoring,verbose=3)
    selector.fit(X_train,y_train)
    selected_columns = X_train.columns[selector.support_]
    removed_columns = X_train.columns[~selector.support_]
    print('*'*20+'SELECTED'+'*'*19)
    Print(list(selected_columns))
    return [selected_columns,selector.grid_scores_]

    
```

######XGBoost Feature-Selection 



```python
xg_final = xgb.XGBClassifier()
xg_selector,grid = rfe_cv(xg_final,1,2,'accuracy', X_tunning_train, y_tunning_train) 
save_feat(xg_selector,"XGBoost") 

X_xg = df[xg_selector]
y_xg = df['y']

Test_xg=Test[xg_selector]

X_xg_train, X_xg_test, y_xg_train, y_xg_test = train_test_split(X_xg, y_xg)
xg_scaler = StandardScaler()  
xg_scaler.fit(X_xg_train)
X_xg_train_scaled = pd.DataFrame(xg_scaler.transform(X_xg_train), columns=X_xg_train.columns) 
X_xg_test_scaled = pd.DataFrame(xg_scaler.transform(X_xg_test), columns=X_xg_test.columns)
Test_xg_selected=pd.DataFrame(xg_scaler.transform(Test_xg), columns=Test_xg.columns)
xg_final_selected=xgboost(X_xg_train_scaled,y_xg_train,X_xg_test_scaled,y_xg_test)

verbose=False 
```

    Fitting estimator with 37 features.
    Fitting estimator with 36 features.
    Fitting estimator with 35 features.
    Fitting estimator with 34 features.
    Fitting estimator with 33 features.
    Fitting estimator with 32 features.
    Fitting estimator with 31 features.
    Fitting estimator with 30 features.
    Fitting estimator with 29 features.
    Fitting estimator with 28 features.
    Fitting estimator with 27 features.
    Fitting estimator with 26 features.
    Fitting estimator with 25 features.
    Fitting estimator with 24 features.
    Fitting estimator with 23 features.
    Fitting estimator with 22 features.
    Fitting estimator with 21 features.
    Fitting estimator with 20 features.
    Fitting estimator with 19 features.
    Fitting estimator with 18 features.
    Fitting estimator with 17 features.
    Fitting estimator with 16 features.
    Fitting estimator with 15 features.
    Fitting estimator with 14 features.
    Fitting estimator with 13 features.
    Fitting estimator with 12 features.
    Fitting estimator with 11 features.
    Fitting estimator with 10 features.
    Fitting estimator with 9 features.
    Fitting estimator with 8 features.
    Fitting estimator with 7 features.
    Fitting estimator with 6 features.
    Fitting estimator with 5 features.
    Fitting estimator with 4 features.
    Fitting estimator with 3 features.
    Fitting estimator with 2 features.
    Fitting estimator with 37 features.
    Fitting estimator with 36 features.
    Fitting estimator with 35 features.
    Fitting estimator with 34 features.
    Fitting estimator with 33 features.
    Fitting estimator with 32 features.
    Fitting estimator with 31 features.
    Fitting estimator with 30 features.
    Fitting estimator with 29 features.
    Fitting estimator with 28 features.
    Fitting estimator with 27 features.
    Fitting estimator with 26 features.
    Fitting estimator with 25 features.
    Fitting estimator with 24 features.
    Fitting estimator with 23 features.
    Fitting estimator with 22 features.
    Fitting estimator with 21 features.
    Fitting estimator with 20 features.
    Fitting estimator with 19 features.
    Fitting estimator with 18 features.
    Fitting estimator with 17 features.
    Fitting estimator with 16 features.
    Fitting estimator with 15 features.
    Fitting estimator with 14 features.
    Fitting estimator with 13 features.
    Fitting estimator with 12 features.
    Fitting estimator with 11 features.
    Fitting estimator with 10 features.
    Fitting estimator with 9 features.
    Fitting estimator with 8 features.
    Fitting estimator with 7 features.
    Fitting estimator with 6 features.
    Fitting estimator with 5 features.
    Fitting estimator with 4 features.
    Fitting estimator with 3 features.
    Fitting estimator with 2 features.
    Fitting estimator with 37 features.
    Fitting estimator with 36 features.
    Fitting estimator with 35 features.
    Fitting estimator with 34 features.
    Fitting estimator with 33 features.
    Fitting estimator with 32 features.
    Fitting estimator with 31 features.
    Fitting estimator with 30 features.
    Fitting estimator with 29 features.
    Fitting estimator with 28 features.
    Fitting estimator with 27 features.
    ********************SELECTED*******************
    ['qs1',
     'qs3',
     'qs4',
     'qs5',
     'qs6',
     'qs10',
     'qs12',
     'qs13',
     'qr1',
     'qr4',
     'qr5',
     'qr9',
     'qr10',
     'qr11',
     'qr12',
     's1',
     's2',
     's3',
     's4',
     's5',
     's6',
     's7',
     's8',
     's9',
     's10',
     's11']


######LightGBM Feature-Selection 



```python
lgbm_final = lgb.LGBMClassifier()
lgbm_selector,grid_lgbm = rfe_cv(lgbm_final,1,2,'accuracy', X_tunning_train, y_tunning_train) 
save_feat(lgbm_selector,"lgbm") 

X_xg = df[lgbm_selector]
y_xg = df['y']

Test_xg=Test[lgbm_selector]

X_xg_train, X_xg_test, y_xg_train, y_xg_test = train_test_split(X_xg, y_xg)
xg_scaler = StandardScaler()  
xg_scaler.fit(X_xg_train)
X_xg_train_scaled = pd.DataFrame(xg_scaler.transform(X_xg_train), columns=X_xg_train.columns) 
X_xg_test_scaled = pd.DataFrame(xg_scaler.transform(X_xg_test), columns=X_xg_test.columns)
Test_xg_selected=pd.DataFrame(xg_scaler.transform(Test_xg), columns=Test_xg.columns)
lgbm_final_selected=lgbm(X_xg_train_scaled,y_xg_train,X_xg_test_scaled,y_xg_test)

verbose=False 


```

    Fitting estimator with 37 features.
    Fitting estimator with 36 features.
    Fitting estimator with 35 features.
    Fitting estimator with 34 features.
    Fitting estimator with 33 features.
    Fitting estimator with 32 features.
    Fitting estimator with 31 features.
    Fitting estimator with 30 features.
    Fitting estimator with 29 features.
    Fitting estimator with 28 features.
    Fitting estimator with 27 features.
    Fitting estimator with 26 features.
    Fitting estimator with 25 features.
    Fitting estimator with 24 features.
    Fitting estimator with 23 features.
    Fitting estimator with 22 features.
    Fitting estimator with 21 features.
    Fitting estimator with 20 features.
    Fitting estimator with 19 features.
    Fitting estimator with 18 features.
    Fitting estimator with 17 features.
    Fitting estimator with 16 features.
    Fitting estimator with 15 features.
    Fitting estimator with 14 features.
    Fitting estimator with 13 features.
    Fitting estimator with 12 features.
    Fitting estimator with 11 features.
    Fitting estimator with 10 features.
    Fitting estimator with 9 features.
    Fitting estimator with 8 features.
    Fitting estimator with 7 features.
    Fitting estimator with 6 features.
    Fitting estimator with 5 features.
    Fitting estimator with 4 features.
    Fitting estimator with 3 features.
    Fitting estimator with 2 features.
    Fitting estimator with 37 features.
    Fitting estimator with 36 features.
    Fitting estimator with 35 features.
    Fitting estimator with 34 features.
    Fitting estimator with 33 features.
    Fitting estimator with 32 features.
    Fitting estimator with 31 features.
    Fitting estimator with 30 features.
    Fitting estimator with 29 features.
    Fitting estimator with 28 features.
    Fitting estimator with 27 features.
    Fitting estimator with 26 features.
    Fitting estimator with 25 features.
    Fitting estimator with 24 features.
    Fitting estimator with 23 features.
    Fitting estimator with 22 features.
    Fitting estimator with 21 features.
    Fitting estimator with 20 features.
    Fitting estimator with 19 features.
    Fitting estimator with 18 features.
    Fitting estimator with 17 features.
    Fitting estimator with 16 features.
    Fitting estimator with 15 features.
    Fitting estimator with 14 features.
    Fitting estimator with 13 features.
    Fitting estimator with 12 features.
    Fitting estimator with 11 features.
    Fitting estimator with 10 features.
    Fitting estimator with 9 features.
    Fitting estimator with 8 features.
    Fitting estimator with 7 features.
    Fitting estimator with 6 features.
    Fitting estimator with 5 features.
    Fitting estimator with 4 features.
    Fitting estimator with 3 features.
    Fitting estimator with 2 features.
    Fitting estimator with 37 features.
    Fitting estimator with 36 features.
    Fitting estimator with 35 features.
    Fitting estimator with 34 features.
    Fitting estimator with 33 features.
    Fitting estimator with 32 features.
    Fitting estimator with 31 features.
    Fitting estimator with 30 features.
    Fitting estimator with 29 features.
    Fitting estimator with 28 features.
    Fitting estimator with 27 features.
    Fitting estimator with 26 features.
    Fitting estimator with 25 features.
    Fitting estimator with 24 features.
    Fitting estimator with 23 features.
    Fitting estimator with 22 features.
    Fitting estimator with 21 features.
    Fitting estimator with 20 features.
    ********************SELECTED*******************
    ['qs4',
     'qs5',
     'qs6',
     'qs7',
     'qs10',
     'qs11',
     'qr4',
     'qr5',
     'qr8',
     'qr9',
     'qr10',
     's1',
     's2',
     's3',
     's4',
     's5',
     's7',
     's9',
     's10']


###### Feature importance results: 


```python
feature_importance=[[grid_lgbm[i],grid[i],X_base_train.columns[i]] for i in range(len(X_base_train.columns))]
feature_importance.sort()
feature_names=[i[2]for i in feature_importance]
importance_lgbm=[i[0]-0.9 for i in feature_importance]
importance_xgb=[i[1]-0.9 for i in feature_importance]
fig = go.Figure()
fig.add_trace(go.Bar(
    x=feature_names,
    y=importance_lgbm,
    name='LGBM Feature Importance ',
    marker_color='LightSkyBlue'
))
fig.add_trace(go.Bar(
    x=feature_names,
    y=importance_xgb,
    name='XGB Feature Importance ',
    marker_color='MediumPurple'
))



fig.update_layout(barmode='group',xaxis_tickangle=-45)
fig.show()

```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="26b2f926-ee67-493d-9a65-62c6afaf0e7d" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("26b2f926-ee67-493d-9a65-62c6afaf0e7d")) {
                    Plotly.newPlot(
                        '26b2f926-ee67-493d-9a65-62c6afaf0e7d',
                        [{"marker": {"color": "LightSkyBlue"}, "name": "LGBM Feature Importance ", "type": "bar", "x": ["qs1", "qs2", "qs3", "qs4", "qs5", "qs6", "qs7", "qs8", "qs10", "qs9", "qs11", "qs12", "qr2", "qr3", "qr10", "qr1", "qr11", "qr12", "qs13", "s5", "s7", "qr9", "qr4", "s10", "s9", "qr13", "s3", "qr7", "qr8", "s11", "s4", "s2", "s1", "s6", "qr5", "s8", "qr6"], "y": [0.06294805802526904, 0.06300421151146463, 0.06323818437061302, 0.06342536265793164, 0.09589143659335508, 0.09654656059897049, 0.09684604585868029, 0.09691155825924191, 0.09743565746373417, 0.0974450163781001, 0.09775386055217594, 0.09776321946654187, 0.09783809078146932, 0.09783809078146932, 0.09786616752456712, 0.09788488535329898, 0.09789424426766491, 0.09789424426766491, 0.09790360318203084, 0.09794103883949457, 0.09794103883949457, 0.09795039775386061, 0.09795975666822643, 0.09795975666822643, 0.09795975666822643, 0.09796911558259236, 0.09796911558259236, 0.09797847449695829, 0.09798783341132422, 0.09798783341132433, 0.09799719232569026, 0.09800655124005608, 0.09801591015442213, 0.09802526906878806, 0.09803462798315388, 0.09804398689751992, 0.09808142255498364]}, {"marker": {"color": "MediumPurple"}, "name": "XGB Feature Importance ", "type": "bar", "x": ["qs1", "qs2", "qs3", "qs4", "qs5", "qs6", "qs7", "qs8", "qs10", "qs9", "qs11", "qs12", "qr2", "qr3", "qr10", "qr1", "qr11", "qr12", "qs13", "s5", "s7", "qr9", "qr4", "s10", "s9", "qr13", "s3", "qr7", "qr8", "s11", "s4", "s2", "s1", "s6", "qr5", "s8", "qr6"], "y": [0.06294805802526904, 0.06300421151146463, 0.06323818437061302, 0.06342536265793164, 0.09589143659335508, 0.09654656059897049, 0.09684604585868029, 0.09691155825924191, 0.09743565746373417, 0.0974450163781001, 0.09775386055217594, 0.09776321946654187, 0.09783809078146932, 0.09783809078146932, 0.09786616752456712, 0.09788488535329898, 0.09789424426766491, 0.09789424426766491, 0.09790360318203084, 0.09794103883949457, 0.09794103883949457, 0.09795039775386061, 0.09795975666822643, 0.09795975666822643, 0.09795975666822643, 0.09796911558259236, 0.09796911558259236, 0.09797847449695829, 0.09798783341132422, 0.09798783341132433, 0.09799719232569026, 0.09800655124005608, 0.09801591015442213, 0.09802526906878806, 0.09803462798315388, 0.09804398689751992, 0.09808142255498364]}],
                        {"barmode": "group", "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "xaxis": {"tickangle": -45}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('26b2f926-ee67-493d-9a65-62c6afaf0e7d');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };

            </script>
        </div>
</body>
</html>


$\implies$ As we can see all the feature except [qs1,qs2,qs3,qs4] are equaly importante for our models. We tried eliminating those variables and reavluating the models , but we didn't improve our perfomances because even though those features are less important than the other , still there is no big  difference in the importnce score (0.99 vs 0.96 ) 
 
$\implies$ That is why we concluded that all the features participate positivly and almost equaly in the classification process 

###HyperPramaterTuning



* Hyperparameter tuning is choosing a set of optimal hyperparameters for the learning algorithm before the learning process begins like penalty in logistic regression and number of estimator in lightGBM.
* Tuning Strategies:
  We will explore two different methods for optimizing hyperparameters:

    * Grid Search
    *  Random Search
* Evaluation Metric: F1 

    * Accuracy can be used when the class distribution is similar while F1-score is a better metric when there are imbalanced classes as in the above case.
    
    * In our case , imbalanced class distribution exists and thus F1-score is a better metric to evaluate our model on.


####GridSearch: 

Grid search is a traditional way to perform hyperparameter optimization. It works by searching exhaustively through a specified subset of hyperparameters.

 + Pros: we try all the combinations and determine the best combination  scoring the best .
 - Cons: computationaly consuming 


```python

def grid_tune(estimator, params, cv, scoring, X_train, y_train, X_test, y_test):
    gs = GridSearchCV(estimator, params, cv=cv, scoring=scoring,verbose=9, n_jobs=1)
    gs.fit(X_train,y_train)    
    print('Training Best Score: ', gs.best_score_, '\n')
    print('Training Best Params:  \n', gs.best_params_, '\n\n')
    print('Training Best Estimator:  \n', gs.best_estimator_, '\n\n')
    return gs.best_params_,gs.best_score_

```

#### RandomSearch: 

Random search differs from grid search mainly in that it searches the specified subset of hyperparameters randomly instead of exhaustively. 

 + Pros:The major benefit being decreased processing time.

- Cons:There is a tradeoff to decreased processing time, however. We aren’t guaranteed to find the optimal combination of hyperparameters.


```python

def random_grid(estimator, params, cv, n_iter, scoring, X_train, y_train, X_test, y_test):
    rs = RandomizedSearchCV(estimator, params, cv=cv, n_iter=n_iter, scoring=scoring,n_jobs=-1,verbose=3) 
    rs.fit(X_train,y_train)
    print('Training Best Score: ', rs.best_score_, '\n')
    print('Training Best Params:  \n', rs.best_params_, '\n\n')
    print('Training Best Estimator:  \n', rs.best_estimator_, '\n\n')
    return rs.best_params_,rs.best_score_
```

$\implies$ We tried Tunning most of the baseline models, because with the wrong Hyperpramters a model performance can be very low, but with the right tunning we can achieve way better results. That is why we will tune most of the models and compare the performances again to chose the best ones based on the F1 score ( we didn't use accuracy because F1 score is more suitable for unbalanced data  )

####LogisticReg Tunning 



```python
start_time=time.time()
logreg = LogisticRegression(max_iter=2000)
solver = {'solver': ['newton-cg','lbfgs','liblinear','sag','saga']}
log_solver,acc = grid_tune(logreg,solver,5,'f1',X_tunning_train, y_tunning_train,X_tunning_test,y_tunning_test)
print("------------Tunning Time {} s------------ ".format(time.time()-start_time))
tuned_model_scores["logreg"]=acc

save_param(log_solver,'log_solver',acc)
```

    Training Best Score:  0.9697506390958892 
    
    Training Best Params:  
     {'solver': 'liblinear'} 
    
    
    Training Best Estimator:  
     LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=2000,
                       multi_class='auto', n_jobs=None, penalty='l2',
                       random_state=None, solver='liblinear', tol=0.0001, verbose=0,
                       warm_start=False) 
    
    



```python
start_time=time.time()
log_params = {'penalty': ['l1','l2'],
              'C': [0.001,0.01,0.1,0.5,1.0],
              'class_weight': ['balanced',None],
              'solver':['liblinear']
             }
log_best,acc = grid_tune(logreg,log_params,5,'f1',X_tunning_train, y_tunning_train,X_tunning_test,y_tunning_test)
print("------------Tunning Time {} s------------ ".format(time.time()-start_time))
tuned_model_scores["logreg"]=acc


save_param(log_best,'log_best',acc)
```

    Training Best Score:  0.9697506390958892 
    
    Training Best Params:  
     {'C': 1.0, 'class_weight': None, 'penalty': 'l2', 'solver': 'liblinear'} 
    
    
    Training Best Estimator:  
     LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=2000,
                       multi_class='auto', n_jobs=None, penalty='l2',
                       random_state=None, solver='liblinear', tol=0.0001, verbose=0,
                       warm_start=False) 


####Ridge Tunning 


```python
start_time=time.time()
ridge= RidgeClassifier()
alpha = {'alpha':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
ridge_best,acc= grid_tune(ridge,alpha,5,'f1',X_tunning_train, y_tunning_train,X_tunning_test,y_tunning_test)
print("------------Tunning Time {} s------------ ".format(time.time()-start_time))
tuned_model_scores["ridge"]=acc


save_param(ridge_best,'ridge_best',acc)
```

    Training Best Score:  0.9034958723761228 
    
    Training Best Params:  
     {'alpha': 0.1} 
    
    
    Training Best Estimator:  
     RidgeClassifier(alpha=0.1, class_weight=None, copy_X=True, fit_intercept=True,
                    max_iter=None, normalize=False, random_state=None,
                    solver='auto', tol=0.001) 
    
    
    ------------Tunning Time 6.864957571029663 s------------ 


####SVM Tunning 



```python
start_time=time.time()
svm_kernel = {'kernel': ['linear','poly','rbf','sigmoid']}
svm_solver,acc = grid_tune(SVC(),svm_kernel,5,'f1',X_tunning_train, y_tunning_train,X_tunning_test,y_tunning_test)
print("------------Tunning Time {} s------------ ".format(time.time()-start_time))
tuned_model_scores["svm"]=acc


save_param(svm_solver,'svm_solver',acc)
```

    Training Best Score:  0.9714150143406011 
    
    Training Best Params:  
     {'kernel': 'poly'} 
    
    
    Training Best Estimator:  
     SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='scale', kernel='poly',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False) 
    
    
    ------------Tunning Time 566.3349184989929 s------------ 



```python
start_time=time.time()
svm_params = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
             'class_weight': [None],
              'kernel': ['poly']
              }
svm_best ,acc= grid_tune(SVC(),svm_params,5,'f1',X_tunning_train, y_tunning_train,X_tunning_test,y_tunning_test)
print("------------Tunning Time {} s------------ ".format(time.time()-start_time))
tuned_model_scores["svm"]=acc


save_param(svm_best,'svm_best',acc)
```

    Fitting 5 folds for each of 25 candidates, totalling 125 fits
    [CV] C=0.1, class_weight=None, gamma=1, kernel=poly ..................


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.


    [CV]  C=0.1, class_weight=None, gamma=1, kernel=poly, score=0.966, total=  17.3s
    [CV] C=0.1, class_weight=None, gamma=1, kernel=poly ..................


    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:   17.3s remaining:    0.0s


    [CV]  C=0.1, class_weight=None, gamma=1, kernel=poly, score=0.967, total=  13.7s
    [CV] C=0.1, class_weight=None, gamma=1, kernel=poly ..................


    [Parallel(n_jobs=1)]: Done   2 out of   2 | elapsed:   31.0s remaining:    0.0s


    [CV]  C=0.1, class_weight=None, gamma=1, kernel=poly, score=0.963, total=  15.9s
    [CV] C=0.1, class_weight=None, gamma=1, kernel=poly ..................


    [Parallel(n_jobs=1)]: Done   3 out of   3 | elapsed:   46.9s remaining:    0.0s


    [CV]  C=0.1, class_weight=None, gamma=1, kernel=poly, score=0.963, total=  18.9s
    [CV] C=0.1, class_weight=None, gamma=1, kernel=poly ..................


    [Parallel(n_jobs=1)]: Done   4 out of   4 | elapsed:  1.1min remaining:    0.0s


    [CV]  C=0.1, class_weight=None, gamma=1, kernel=poly, score=0.958, total=  17.7s
    [CV] C=0.1, class_weight=None, gamma=0.1, kernel=poly ................


    [Parallel(n_jobs=1)]: Done   5 out of   5 | elapsed:  1.4min remaining:    0.0s


    [CV]  C=0.1, class_weight=None, gamma=0.1, kernel=poly, score=0.967, total=   8.1s
    [CV] C=0.1, class_weight=None, gamma=0.1, kernel=poly ................


    [Parallel(n_jobs=1)]: Done   6 out of   6 | elapsed:  1.5min remaining:    0.0s


    [CV]  C=0.1, class_weight=None, gamma=0.1, kernel=poly, score=0.970, total=   7.2s
    [CV] C=0.1, class_weight=None, gamma=0.1, kernel=poly ................


    [Parallel(n_jobs=1)]: Done   7 out of   7 | elapsed:  1.6min remaining:    0.0s


    [CV]  C=0.1, class_weight=None, gamma=0.1, kernel=poly, score=0.969, total=   6.9s
    [CV] C=0.1, class_weight=None, gamma=0.1, kernel=poly ................


    [Parallel(n_jobs=1)]: Done   8 out of   8 | elapsed:  1.8min remaining:    0.0s


    [CV]  C=0.1, class_weight=None, gamma=0.1, kernel=poly, score=0.974, total=   7.8s
    [CV] C=0.1, class_weight=None, gamma=0.1, kernel=poly ................
    [CV]  C=0.1, class_weight=None, gamma=0.1, kernel=poly, score=0.962, total=   7.6s
    [CV] C=0.1, class_weight=None, gamma=0.01, kernel=poly ...............
    [CV]  C=0.1, class_weight=None, gamma=0.01, kernel=poly, score=0.951, total=   6.9s
    [CV] C=0.1, class_weight=None, gamma=0.01, kernel=poly ...............
    [CV]  C=0.1, class_weight=None, gamma=0.01, kernel=poly, score=0.956, total=   6.9s
    [CV] C=0.1, class_weight=None, gamma=0.01, kernel=poly ...............
    [CV]  C=0.1, class_weight=None, gamma=0.01, kernel=poly, score=0.947, total=   6.8s
    [CV] C=0.1, class_weight=None, gamma=0.01, kernel=poly ...............
    [CV]  C=0.1, class_weight=None, gamma=0.01, kernel=poly, score=0.953, total=   6.8s
    [CV] C=0.1, class_weight=None, gamma=0.01, kernel=poly ...............
    [CV]  C=0.1, class_weight=None, gamma=0.01, kernel=poly, score=0.953, total=   6.9s
    [CV] C=0.1, class_weight=None, gamma=0.001, kernel=poly ..............
    [CV]  C=0.1, class_weight=None, gamma=0.001, kernel=poly, score=0.631, total=  28.9s
    [CV] C=0.1, class_weight=None, gamma=0.001, kernel=poly ..............
    [CV]  C=0.1, class_weight=None, gamma=0.001, kernel=poly, score=0.664, total=  29.9s
    [CV] C=0.1, class_weight=None, gamma=0.001, kernel=poly ..............
    [CV]  C=0.1, class_weight=None, gamma=0.001, kernel=poly, score=0.646, total=  29.5s
    [CV] C=0.1, class_weight=None, gamma=0.001, kernel=poly ..............
    [CV]  C=0.1, class_weight=None, gamma=0.001, kernel=poly, score=0.646, total=  29.5s
    [CV] C=0.1, class_weight=None, gamma=0.001, kernel=poly ..............
    [CV]  C=0.1, class_weight=None, gamma=0.001, kernel=poly, score=0.666, total=  29.6s
    [CV] C=0.1, class_weight=None, gamma=0.0001, kernel=poly .............
    [CV]  C=0.1, class_weight=None, gamma=0.0001, kernel=poly, score=0.000, total=  39.5s
    [CV] C=0.1, class_weight=None, gamma=0.0001, kernel=poly .............
    [CV]  C=0.1, class_weight=None, gamma=0.0001, kernel=poly, score=0.000, total=  39.3s
    [CV] C=0.1, class_weight=None, gamma=0.0001, kernel=poly .............
    [CV]  C=0.1, class_weight=None, gamma=0.0001, kernel=poly, score=0.000, total=  39.5s
    [CV] C=0.1, class_weight=None, gamma=0.0001, kernel=poly .............
    [CV]  C=0.1, class_weight=None, gamma=0.0001, kernel=poly, score=0.000, total=  40.6s
    [CV] C=0.1, class_weight=None, gamma=0.0001, kernel=poly .............
    [CV]  C=0.1, class_weight=None, gamma=0.0001, kernel=poly, score=0.000, total=  40.4s
    [CV] C=1, class_weight=None, gamma=1, kernel=poly ....................
    [CV]  C=1, class_weight=None, gamma=1, kernel=poly, score=0.966, total=  17.5s
    [CV] C=1, class_weight=None, gamma=1, kernel=poly ....................
    [CV]  C=1, class_weight=None, gamma=1, kernel=poly, score=0.967, total=  14.1s
    [CV] C=1, class_weight=None, gamma=1, kernel=poly ....................
    [CV]  C=1, class_weight=None, gamma=1, kernel=poly, score=0.963, total=  16.1s
    [CV] C=1, class_weight=None, gamma=1, kernel=poly ....................
    [CV]  C=1, class_weight=None, gamma=1, kernel=poly, score=0.963, total=  19.0s
    [CV] C=1, class_weight=None, gamma=1, kernel=poly ....................
    [CV]  C=1, class_weight=None, gamma=1, kernel=poly, score=0.958, total=  18.5s
    [CV] C=1, class_weight=None, gamma=0.1, kernel=poly ..................
    [CV]  C=1, class_weight=None, gamma=0.1, kernel=poly, score=0.967, total=  17.9s
    [CV] C=1, class_weight=None, gamma=0.1, kernel=poly ..................
    [CV]  C=1, class_weight=None, gamma=0.1, kernel=poly, score=0.965, total=  16.5s
    [CV] C=1, class_weight=None, gamma=0.1, kernel=poly ..................
    [CV]  C=1, class_weight=None, gamma=0.1, kernel=poly, score=0.962, total=  15.9s
    [CV] C=1, class_weight=None, gamma=0.1, kernel=poly ..................
    [CV]  C=1, class_weight=None, gamma=0.1, kernel=poly, score=0.965, total=  21.9s
    [CV] C=1, class_weight=None, gamma=0.1, kernel=poly ..................
    [CV]  C=1, class_weight=None, gamma=0.1, kernel=poly, score=0.956, total=  19.8s
    [CV] C=1, class_weight=None, gamma=0.01, kernel=poly .................
    [CV]  C=1, class_weight=None, gamma=0.01, kernel=poly, score=0.964, total=   5.7s
    [CV] C=1, class_weight=None, gamma=0.01, kernel=poly .................
    [CV]  C=1, class_weight=None, gamma=0.01, kernel=poly, score=0.965, total=   5.3s
    [CV] C=1, class_weight=None, gamma=0.01, kernel=poly .................
    [CV]  C=1, class_weight=None, gamma=0.01, kernel=poly, score=0.962, total=   5.2s
    [CV] C=1, class_weight=None, gamma=0.01, kernel=poly .................
    [CV]  C=1, class_weight=None, gamma=0.01, kernel=poly, score=0.968, total=   5.5s
    [CV] C=1, class_weight=None, gamma=0.01, kernel=poly .................
    [CV]  C=1, class_weight=None, gamma=0.01, kernel=poly, score=0.962, total=   5.3s
    [CV] C=1, class_weight=None, gamma=0.001, kernel=poly ................
    [CV]  C=1, class_weight=None, gamma=0.001, kernel=poly, score=0.836, total=  18.3s
    [CV] C=1, class_weight=None, gamma=0.001, kernel=poly ................
    [CV]  C=1, class_weight=None, gamma=0.001, kernel=poly, score=0.840, total=  18.7s
    [CV] C=1, class_weight=None, gamma=0.001, kernel=poly ................
    [CV]  C=1, class_weight=None, gamma=0.001, kernel=poly, score=0.832, total=  18.8s
    [CV] C=1, class_weight=None, gamma=0.001, kernel=poly ................
    [CV]  C=1, class_weight=None, gamma=0.001, kernel=poly, score=0.838, total=  18.8s
    [CV] C=1, class_weight=None, gamma=0.001, kernel=poly ................
    [CV]  C=1, class_weight=None, gamma=0.001, kernel=poly, score=0.839, total=  18.9s
    [CV] C=1, class_weight=None, gamma=0.0001, kernel=poly ...............
    [CV]  C=1, class_weight=None, gamma=0.0001, kernel=poly, score=0.000, total=  41.1s
    [CV] C=1, class_weight=None, gamma=0.0001, kernel=poly ...............
    [CV]  C=1, class_weight=None, gamma=0.0001, kernel=poly, score=0.000, total=  41.4s
    [CV] C=1, class_weight=None, gamma=0.0001, kernel=poly ...............
    [CV]  C=1, class_weight=None, gamma=0.0001, kernel=poly, score=0.000, total=  40.8s
    [CV] C=1, class_weight=None, gamma=0.0001, kernel=poly ...............
    [CV]  C=1, class_weight=None, gamma=0.0001, kernel=poly, score=0.000, total=  41.0s
    [CV] C=1, class_weight=None, gamma=0.0001, kernel=poly ...............
    [CV]  C=1, class_weight=None, gamma=0.0001, kernel=poly, score=0.000, total=  40.7s
    [CV] C=10, class_weight=None, gamma=1, kernel=poly ...................
    [CV]  C=10, class_weight=None, gamma=1, kernel=poly, score=0.966, total=  17.6s
    [CV] C=10, class_weight=None, gamma=1, kernel=poly ...................
    [CV]  C=10, class_weight=None, gamma=1, kernel=poly, score=0.967, total=  14.1s
    [CV] C=10, class_weight=None, gamma=1, kernel=poly ...................
    [CV]  C=10, class_weight=None, gamma=1, kernel=poly, score=0.963, total=  16.0s
    [CV] C=10, class_weight=None, gamma=1, kernel=poly ...................
    [CV]  C=10, class_weight=None, gamma=1, kernel=poly, score=0.963, total=  18.7s
    [CV] C=10, class_weight=None, gamma=1, kernel=poly ...................
    [CV]  C=10, class_weight=None, gamma=1, kernel=poly, score=0.958, total=  17.7s
    [CV] C=10, class_weight=None, gamma=0.1, kernel=poly .................
    [CV]  C=10, class_weight=None, gamma=0.1, kernel=poly, score=0.966, total=  15.6s
    [CV] C=10, class_weight=None, gamma=0.1, kernel=poly .................
    [CV]  C=10, class_weight=None, gamma=0.1, kernel=poly, score=0.967, total=  11.8s
    [CV] C=10, class_weight=None, gamma=0.1, kernel=poly .................
    [CV]  C=10, class_weight=None, gamma=0.1, kernel=poly, score=0.963, total=  17.2s
    [CV] C=10, class_weight=None, gamma=0.1, kernel=poly .................
    [CV]  C=10, class_weight=None, gamma=0.1, kernel=poly, score=0.963, total=  24.2s
    [CV] C=10, class_weight=None, gamma=0.1, kernel=poly .................
    [CV]  C=10, class_weight=None, gamma=0.1, kernel=poly, score=0.958, total=  15.4s
    [CV] C=10, class_weight=None, gamma=0.01, kernel=poly ................
    [CV]  C=10, class_weight=None, gamma=0.01, kernel=poly, score=0.971, total=   5.7s
    [CV] C=10, class_weight=None, gamma=0.01, kernel=poly ................
    [CV]  C=10, class_weight=None, gamma=0.01, kernel=poly, score=0.975, total=   5.1s
    [CV] C=10, class_weight=None, gamma=0.01, kernel=poly ................
    [CV]  C=10, class_weight=None, gamma=0.01, kernel=poly, score=0.966, total=   5.1s
    [CV] C=10, class_weight=None, gamma=0.01, kernel=poly ................
    [CV]  C=10, class_weight=None, gamma=0.01, kernel=poly, score=0.976, total=   5.7s
    [CV] C=10, class_weight=None, gamma=0.01, kernel=poly ................
    [CV]  C=10, class_weight=None, gamma=0.01, kernel=poly, score=0.968, total=   5.0s
    [CV] C=10, class_weight=None, gamma=0.001, kernel=poly ...............
    [CV]  C=10, class_weight=None, gamma=0.001, kernel=poly, score=0.923, total=  10.9s
    [CV] C=10, class_weight=None, gamma=0.001, kernel=poly ...............
    [CV]  C=10, class_weight=None, gamma=0.001, kernel=poly, score=0.923, total=  10.9s
    [CV] C=10, class_weight=None, gamma=0.001, kernel=poly ...............
    [CV]  C=10, class_weight=None, gamma=0.001, kernel=poly, score=0.920, total=  10.9s
    [CV] C=10, class_weight=None, gamma=0.001, kernel=poly ...............
    [CV]  C=10, class_weight=None, gamma=0.001, kernel=poly, score=0.920, total=  10.9s
    [CV] C=10, class_weight=None, gamma=0.001, kernel=poly ...............
    [CV]  C=10, class_weight=None, gamma=0.001, kernel=poly, score=0.922, total=  10.8s
    [CV] C=10, class_weight=None, gamma=0.0001, kernel=poly ..............
    [CV]  C=10, class_weight=None, gamma=0.0001, kernel=poly, score=0.169, total=  39.3s
    [CV] C=10, class_weight=None, gamma=0.0001, kernel=poly ..............
    [CV]  C=10, class_weight=None, gamma=0.0001, kernel=poly, score=0.236, total=  39.8s
    [CV] C=10, class_weight=None, gamma=0.0001, kernel=poly ..............
    [CV]  C=10, class_weight=None, gamma=0.0001, kernel=poly, score=0.200, total=  39.4s
    [CV] C=10, class_weight=None, gamma=0.0001, kernel=poly ..............
    [CV]  C=10, class_weight=None, gamma=0.0001, kernel=poly, score=0.208, total=  39.1s
    [CV] C=10, class_weight=None, gamma=0.0001, kernel=poly ..............
    [CV]  C=10, class_weight=None, gamma=0.0001, kernel=poly, score=0.218, total=  39.3s
    [CV] C=100, class_weight=None, gamma=1, kernel=poly ..................
    [CV]  C=100, class_weight=None, gamma=1, kernel=poly, score=0.966, total=  17.2s
    [CV] C=100, class_weight=None, gamma=1, kernel=poly ..................
    [CV]  C=100, class_weight=None, gamma=1, kernel=poly, score=0.967, total=  13.9s
    [CV] C=100, class_weight=None, gamma=1, kernel=poly ..................
    [CV]  C=100, class_weight=None, gamma=1, kernel=poly, score=0.963, total=  15.7s
    [CV] C=100, class_weight=None, gamma=1, kernel=poly ..................
    [CV]  C=100, class_weight=None, gamma=1, kernel=poly, score=0.963, total=  18.6s
    [CV] C=100, class_weight=None, gamma=1, kernel=poly ..................
    [CV]  C=100, class_weight=None, gamma=1, kernel=poly, score=0.958, total=  17.8s
    [CV] C=100, class_weight=None, gamma=0.1, kernel=poly ................
    [CV]  C=100, class_weight=None, gamma=0.1, kernel=poly, score=0.966, total=  15.5s
    [CV] C=100, class_weight=None, gamma=0.1, kernel=poly ................
    [CV]  C=100, class_weight=None, gamma=0.1, kernel=poly, score=0.967, total=  12.2s
    [CV] C=100, class_weight=None, gamma=0.1, kernel=poly ................
    [CV]  C=100, class_weight=None, gamma=0.1, kernel=poly, score=0.963, total=  16.9s
    [CV] C=100, class_weight=None, gamma=0.1, kernel=poly ................
    [CV]  C=100, class_weight=None, gamma=0.1, kernel=poly, score=0.963, total=  23.2s
    [CV] C=100, class_weight=None, gamma=0.1, kernel=poly ................
    [CV]  C=100, class_weight=None, gamma=0.1, kernel=poly, score=0.958, total=  15.4s
    [CV] C=100, class_weight=None, gamma=0.01, kernel=poly ...............
    [CV]  C=100, class_weight=None, gamma=0.01, kernel=poly, score=0.967, total=   8.5s
    [CV] C=100, class_weight=None, gamma=0.01, kernel=poly ...............
    [CV]  C=100, class_weight=None, gamma=0.01, kernel=poly, score=0.970, total=   7.0s
    [CV] C=100, class_weight=None, gamma=0.01, kernel=poly ...............
    [CV]  C=100, class_weight=None, gamma=0.01, kernel=poly, score=0.969, total=   8.1s
    [CV] C=100, class_weight=None, gamma=0.01, kernel=poly ...............
    [CV]  C=100, class_weight=None, gamma=0.01, kernel=poly, score=0.974, total=   8.1s
    [CV] C=100, class_weight=None, gamma=0.01, kernel=poly ...............
    [CV]  C=100, class_weight=None, gamma=0.01, kernel=poly, score=0.962, total=   7.8s
    [CV] C=100, class_weight=None, gamma=0.001, kernel=poly ..............
    [CV]  C=100, class_weight=None, gamma=0.001, kernel=poly, score=0.951, total=   7.0s
    [CV] C=100, class_weight=None, gamma=0.001, kernel=poly ..............
    [CV]  C=100, class_weight=None, gamma=0.001, kernel=poly, score=0.956, total=   7.0s
    [CV] C=100, class_weight=None, gamma=0.001, kernel=poly ..............
    [CV]  C=100, class_weight=None, gamma=0.001, kernel=poly, score=0.947, total=   6.9s
    [CV] C=100, class_weight=None, gamma=0.001, kernel=poly ..............
    [CV]  C=100, class_weight=None, gamma=0.001, kernel=poly, score=0.953, total=   6.9s
    [CV] C=100, class_weight=None, gamma=0.001, kernel=poly ..............
    [CV]  C=100, class_weight=None, gamma=0.001, kernel=poly, score=0.953, total=   7.0s
    [CV] C=100, class_weight=None, gamma=0.0001, kernel=poly .............
    [CV]  C=100, class_weight=None, gamma=0.0001, kernel=poly, score=0.631, total=  29.2s
    [CV] C=100, class_weight=None, gamma=0.0001, kernel=poly .............
    [CV]  C=100, class_weight=None, gamma=0.0001, kernel=poly, score=0.664, total=  30.6s
    [CV] C=100, class_weight=None, gamma=0.0001, kernel=poly .............
    [CV]  C=100, class_weight=None, gamma=0.0001, kernel=poly, score=0.646, total=  29.9s
    [CV] C=100, class_weight=None, gamma=0.0001, kernel=poly .............
    [CV]  C=100, class_weight=None, gamma=0.0001, kernel=poly, score=0.646, total=  30.1s
    [CV] C=100, class_weight=None, gamma=0.0001, kernel=poly .............
    [CV]  C=100, class_weight=None, gamma=0.0001, kernel=poly, score=0.666, total=  30.2s
    [CV] C=1000, class_weight=None, gamma=1, kernel=poly .................
    [CV]  C=1000, class_weight=None, gamma=1, kernel=poly, score=0.966, total=  17.6s
    [CV] C=1000, class_weight=None, gamma=1, kernel=poly .................
    [CV]  C=1000, class_weight=None, gamma=1, kernel=poly, score=0.967, total=  14.4s
    [CV] C=1000, class_weight=None, gamma=1, kernel=poly .................
    [CV]  C=1000, class_weight=None, gamma=1, kernel=poly, score=0.963, total=  15.9s
    [CV] C=1000, class_weight=None, gamma=1, kernel=poly .................
    [CV]  C=1000, class_weight=None, gamma=1, kernel=poly, score=0.963, total=  18.8s
    [CV] C=1000, class_weight=None, gamma=1, kernel=poly .................
    [CV]  C=1000, class_weight=None, gamma=1, kernel=poly, score=0.958, total=  18.0s
    [CV] C=1000, class_weight=None, gamma=0.1, kernel=poly ...............
    [CV]  C=1000, class_weight=None, gamma=0.1, kernel=poly, score=0.966, total=  15.5s
    [CV] C=1000, class_weight=None, gamma=0.1, kernel=poly ...............
    [CV]  C=1000, class_weight=None, gamma=0.1, kernel=poly, score=0.967, total=  11.7s
    [CV] C=1000, class_weight=None, gamma=0.1, kernel=poly ...............
    [CV]  C=1000, class_weight=None, gamma=0.1, kernel=poly, score=0.963, total=  16.9s
    [CV] C=1000, class_weight=None, gamma=0.1, kernel=poly ...............
    [CV]  C=1000, class_weight=None, gamma=0.1, kernel=poly, score=0.963, total=  24.1s
    [CV] C=1000, class_weight=None, gamma=0.1, kernel=poly ...............
    [CV]  C=1000, class_weight=None, gamma=0.1, kernel=poly, score=0.958, total=  15.5s
    [CV] C=1000, class_weight=None, gamma=0.01, kernel=poly ..............
    [CV]  C=1000, class_weight=None, gamma=0.01, kernel=poly, score=0.967, total=  15.9s
    [CV] C=1000, class_weight=None, gamma=0.01, kernel=poly ..............
    [CV]  C=1000, class_weight=None, gamma=0.01, kernel=poly, score=0.965, total=  12.2s
    [CV] C=1000, class_weight=None, gamma=0.01, kernel=poly ..............
    [CV]  C=1000, class_weight=None, gamma=0.01, kernel=poly, score=0.962, total=  15.5s
    [CV] C=1000, class_weight=None, gamma=0.01, kernel=poly ..............
    [CV]  C=1000, class_weight=None, gamma=0.01, kernel=poly, score=0.965, total=  22.2s
    [CV] C=1000, class_weight=None, gamma=0.01, kernel=poly ..............
    [CV]  C=1000, class_weight=None, gamma=0.01, kernel=poly, score=0.956, total=  16.5s
    [CV] C=1000, class_weight=None, gamma=0.001, kernel=poly .............
    [CV]  C=1000, class_weight=None, gamma=0.001, kernel=poly, score=0.964, total=   5.7s
    [CV] C=1000, class_weight=None, gamma=0.001, kernel=poly .............
    [CV]  C=1000, class_weight=None, gamma=0.001, kernel=poly, score=0.965, total=   5.4s
    [CV] C=1000, class_weight=None, gamma=0.001, kernel=poly .............
    [CV]  C=1000, class_weight=None, gamma=0.001, kernel=poly, score=0.962, total=   5.1s
    [CV] C=1000, class_weight=None, gamma=0.001, kernel=poly .............
    [CV]  C=1000, class_weight=None, gamma=0.001, kernel=poly, score=0.968, total=   5.5s
    [CV] C=1000, class_weight=None, gamma=0.001, kernel=poly .............
    [CV]  C=1000, class_weight=None, gamma=0.001, kernel=poly, score=0.962, total=   5.1s
    [CV] C=1000, class_weight=None, gamma=0.0001, kernel=poly ............
    [CV]  C=1000, class_weight=None, gamma=0.0001, kernel=poly, score=0.836, total=  18.1s
    [CV] C=1000, class_weight=None, gamma=0.0001, kernel=poly ............
    [CV]  C=1000, class_weight=None, gamma=0.0001, kernel=poly, score=0.840, total=  18.5s
    [CV] C=1000, class_weight=None, gamma=0.0001, kernel=poly ............
    [CV]  C=1000, class_weight=None, gamma=0.0001, kernel=poly, score=0.832, total=  18.3s
    [CV] C=1000, class_weight=None, gamma=0.0001, kernel=poly ............
    [CV]  C=1000, class_weight=None, gamma=0.0001, kernel=poly, score=0.838, total=  18.7s
    [CV] C=1000, class_weight=None, gamma=0.0001, kernel=poly ............
    [CV]  C=1000, class_weight=None, gamma=0.0001, kernel=poly, score=0.839, total=  18.2s


    [Parallel(n_jobs=1)]: Done 125 out of 125 | elapsed: 36.9min finished


    Training Best Score:  0.9712217256644158 
    
    Training Best Params:  
     {'C': 10, 'class_weight': None, 'gamma': 0.01, 'kernel': 'poly'} 
    
    
    Training Best Estimator:  
     SVC(C=10, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma=0.01, kernel='poly',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False) 
    
    
    ------------Tunning Time 2219.7143037319183 s------------ 



    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-18-48a589f34df0> in <module>()
          7 svm_best ,acc= grid_tune(SVC(),svm_params,5,'f1',X_tunning_train, y_tunning_train,X_tunning_test,y_tunning_test)
          8 print("------------Tunning Time {} s------------ ".format(time.time()-start_time))
    ----> 9 save_param(svm_best,'svm_best',acc)
    

    NameError: name 'save_param' is not defined


####KNN Tunning 


```python
start_time=time.time()
knn = KNeighborsClassifier()
knn_params={
    'n_neighbors' : range(1, 21, 2),
    }
knn_best,acc = grid_tune(knn,knn_params,3,'accuracy',X_tunning_train, y_tunning_train,X_tunning_test,y_tunning_test)
print("------------Tunning Time {} s------------ ".format(time.time()-start_time))
tuned_model_scores["knn"]=acc



save_param(knn_best,'knn_best',acc)
```

    Fitting 3 folds for each of 10 candidates, totalling 30 fits
    [CV] n_neighbors=1 ...................................................


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.


    [CV] ....................... n_neighbors=1, score=0.997, total= 5.6min
    [CV] n_neighbors=1 ...................................................


    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:  5.6min remaining:    0.0s


    [CV] ....................... n_neighbors=1, score=0.997, total= 6.0min
    [CV] n_neighbors=1 ...................................................


    [Parallel(n_jobs=1)]: Done   2 out of   2 | elapsed: 11.6min remaining:    0.0s


    [CV] ....................... n_neighbors=1, score=0.997, total= 6.1min
    [CV] n_neighbors=3 ...................................................


    [Parallel(n_jobs=1)]: Done   3 out of   3 | elapsed: 17.7min remaining:    0.0s


    [CV] ....................... n_neighbors=3, score=0.997, total= 6.3min
    [CV] n_neighbors=3 ...................................................


    [Parallel(n_jobs=1)]: Done   4 out of   4 | elapsed: 24.0min remaining:    0.0s


    [CV] ....................... n_neighbors=3, score=0.997, total= 5.7min
    [CV] n_neighbors=3 ...................................................


    [Parallel(n_jobs=1)]: Done   5 out of   5 | elapsed: 29.8min remaining:    0.0s


    [CV] ....................... n_neighbors=3, score=0.997, total= 5.4min
    [CV] n_neighbors=5 ...................................................


    [Parallel(n_jobs=1)]: Done   6 out of   6 | elapsed: 35.1min remaining:    0.0s


    [CV] ....................... n_neighbors=5, score=0.997, total= 5.3min
    [CV] n_neighbors=5 ...................................................


    [Parallel(n_jobs=1)]: Done   7 out of   7 | elapsed: 40.5min remaining:    0.0s


    [CV] ....................... n_neighbors=5, score=0.997, total= 5.0min
    [CV] n_neighbors=5 ...................................................


    [Parallel(n_jobs=1)]: Done   8 out of   8 | elapsed: 45.5min remaining:    0.0s


    [CV] ....................... n_neighbors=5, score=0.997, total= 5.0min
    [CV] n_neighbors=7 ...................................................
    [CV] ....................... n_neighbors=7, score=0.997, total= 5.5min
    [CV] n_neighbors=7 ...................................................
    [CV] ....................... n_neighbors=7, score=0.997, total= 5.5min
    [CV] n_neighbors=7 ...................................................
    [CV] ....................... n_neighbors=7, score=0.997, total= 5.3min
    [CV] n_neighbors=9 ...................................................
    [CV] ....................... n_neighbors=9, score=0.997, total= 5.2min
    [CV] n_neighbors=9 ...................................................
    [CV] ....................... n_neighbors=9, score=0.997, total= 5.0min
    [CV] n_neighbors=9 ...................................................
    [CV] ....................... n_neighbors=9, score=0.996, total= 4.9min
    [CV] n_neighbors=11 ..................................................
    [CV] ...................... n_neighbors=11, score=0.996, total= 5.0min
    [CV] n_neighbors=11 ..................................................
    [CV] ...................... n_neighbors=11, score=0.997, total= 5.0min
    [CV] n_neighbors=11 ..................................................
    [CV] ...................... n_neighbors=11, score=0.996, total= 5.1min
    [CV] n_neighbors=13 ..................................................
    [CV] ...................... n_neighbors=13, score=0.996, total= 5.0min
    [CV] n_neighbors=13 ..................................................
    [CV] ...................... n_neighbors=13, score=0.996, total= 4.9min
    [CV] n_neighbors=13 ..................................................
    [CV] ...................... n_neighbors=13, score=0.996, total= 5.0min
    [CV] n_neighbors=15 ..................................................
    [CV] ...................... n_neighbors=15, score=0.996, total= 5.1min
    [CV] n_neighbors=15 ..................................................
    [CV] ...................... n_neighbors=15, score=0.997, total= 5.1min
    [CV] n_neighbors=15 ..................................................
    [CV] ...................... n_neighbors=15, score=0.996, total= 5.4min
    [CV] n_neighbors=17 ..................................................
    [CV] ...................... n_neighbors=17, score=0.996, total= 5.9min
    [CV] n_neighbors=17 ..................................................
    [CV] ...................... n_neighbors=17, score=0.996, total= 5.4min
    [CV] n_neighbors=17 ..................................................
    [CV] ...................... n_neighbors=17, score=0.996, total= 5.6min
    [CV] n_neighbors=19 ..................................................
    [CV] ...................... n_neighbors=19, score=0.996, total= 5.8min
    [CV] n_neighbors=19 ..................................................
    [CV] ...................... n_neighbors=19, score=0.996, total= 5.6min
    [CV] n_neighbors=19 ..................................................
    [CV] ...................... n_neighbors=19, score=0.996, total= 5.8min


    [Parallel(n_jobs=1)]: Done  30 out of  30 | elapsed: 161.5min finished


    Training Best Score:  0.9969489906753376 
    
    Training Best Params:  
     {'n_neighbors': 1} 
    
    
    Training Best Estimator:  
     KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=None, n_neighbors=1, p=2,
                         weights='uniform') 
    
    
    ------------Tunning Time 9718.721629619598 s------------ 


####RandomForest Tunnning


```python
start_time=time.time()
rf = RandomForestClassifier()
rf_params = {'n_estimators': np.arange(10,300,10)}
rf_est,acc = grid_tune(rf,rf_params,5,'f1',X_tunning_train, y_tunning_train,X_tunning_test,y_tunning_test)
print("------------Tunning Time {} s------------ ".format(time.time()-start_time))
tuned_model_scores["logreg"]=acc

save_param(rf_est,'rf_est',acc)
```

    Fitting 5 folds for each of 29 candidates, totalling 145 fits
    
    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    
    [Parallel(n_jobs=1)]: Done   24 out of   24 | elapsed:    20.4s remaining:    0.0s
    
    [Parallel(n_jobs=1)]: Done   72 out of   72 | elapsed:    80.8s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   145 out of   145 | elapsed:   164.2s finished
    
    Training Best Score:  0.9731638186504152 
    
    Training Best Params:  
     {'n_estimators': 35} 
    
    
    Training Best Estimator:  
     RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=None, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=35,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False) 
    
    
    ------------Tunning Time 150.6553213596344 s------------



```python
start_time=time.time()
rf_params = {'min_samples_split': np.arange(2,50,4),
            'n_estimators': [35]} ## from previous tunning 
rf_split,acc = grid_tune(rf,rf_params,5,'f1',X_tunning_train, y_tunning_train,X_tunning_test,y_tunning_test)
print("------------Tunning Time {} s------------ ".format(time.time()-start_time))
tuned_model_scores["rf"]=acc

save_param(rf_split,'rf_split',acc)
```

    
    Fitting 5 folds for each of 11 candidates, totalling 55 fits
    
    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    
    [Parallel(n_jobs=1)]: Done   14 out of   14 | elapsed:    25.5s remaining:    0.0s
    
    [Parallel(n_jobs=1)]: Done   55 out of   55 | elapsed:   64.1s finished
    
    Training Best Score:  0.988521486504152 
    
    Training Best Params:  
     {'n_estimators': 35,'min_samples_split': 14 } 
    
    
    Training Best Estimator:  
     RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=None, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=14,
                           min_weight_fraction_leaf=0.0, n_estimators=35,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False) 
    
    
    ------------Tunning Time 65.2532882244 s------------ 
    



```python
start_time=time.time()
rf_params = {'max_depth': np.arange(10,60,1),
                   'min_samples_split': [14],
                   'n_estimators': [35]}
rf_depth ,acc= grid_tune(rf,rf_params,5,'f1',X_tunning_train, y_tunning_train,X_tunning_test,y_tunning_test)
print("------------Tunning Time {} s------------ ".format(time.time()-start_time))
tuned_model_scores["rf"]=acc

save_param(rf_depth,'rf_depth',acc)
```

    Fitting 5 folds for each of 59 candidates, totalling 295 fits
    
    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    
    [Parallel(n_jobs=1)]: Done   52 out of   50 | elapsed:    82.5s remaining:    0.0s
    
    [Parallel(n_jobs=1)]: Done   157 out of   157 | elapsed:    261.9s remaining:    0.0s
    
    [Parallel(n_jobs=1)]: Done   295 out of   295 | elapsed:  510.1s finished
    
    Training Best Score:  0.998521486504152 
    
    Training Best Params:  
     {'n_estimators': 35,'min_samples_split': 14,'max_depth': 34, } 
    
    
    Training Best Estimator:  
     RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=34, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=14,
                           min_weight_fraction_leaf=0.0, n_estimators=35,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False) 
    
    
    ------------Tunning Time 511.7858521882244 s------------ 


we can  expend our search space without making the exectution time very long by using Randomsearch insted of gridsearch.

$\implies$ As you can see we got diffrent set of hyperparamters, we are not 100% sure they are the optimal set of paramters but still with a high number of iteration we can achive good results (Gridsearch 0.9985 vs RandomSearch 0.99846 ) 


```python
start_time=time.time()
rf_params = {'max_depth': np.arange(10,60,1),
             'min_samples_split': np.arange(2,50,1),
             'n_estimators': np.arange(10,1000,5)}
rf_rg,acc= random_grid(rf,rf_params,5,50,'f1',X_tunning_train, y_tunning_train,X_tunning_test,y_tunning_test)
print("------------Tunning Time {} s------------ ".format(time.time()-start_time))
tuned_model_scores["rf"]=acc

save_param(rf_rg,'rf_rg',acc)


```

    Fitting 5 folds for each of 50 candidates, totalling 250 fits
    
    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  24 tasks      | elapsed:  6.3min
    [Parallel(n_jobs=-1)]: Done 120 tasks      | elapsed: 18.7min
    [Parallel(n_jobs=-1)]: Done 250 out of 250 | elapsed: 58.1min finished
    
    Training Best Score:  0.998462827182174 
    
    Training Best Params:  
     {'n_estimators': 15,'min_samples_split': 7,'max_depth': 55 } 
    
    Training Best Estimator:  
    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=55, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=7,
                           min_weight_fraction_leaf=0.0, n_estimators=15,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False) 
    
    
    
    ------------Tunning Time 3630.7181183510284 s------------ 
    


####XGBoost Tunning 


```python
start_time=time.time()
xg = xgb.XGBClassifier()
xg_params= {'eta': [0.01,0.05,0.1,0.2,0.3],
             'min_depth': np.arange(3,10,1),
             'min_child_weight': np.arange(1,6,1),
             'scale_pos_weight': [0.5,1,2],
             'objective': ['binary:logistic', 'binary:logitraw','binary:hinge']
            }
xg_rg ,acc= random_grid(xg,xg_params,4,60,'accuracy',X_tunning_train, y_tunning_train,X_tunning_test,y_tunning_test)
print("------------Tunning Time {} s------------ ".format(time.time()-start_time))
tuned_model_scores["xgb"]=acc


save_param(xg_rg,'xg_rg',acc)
```

    Fitting 4 folds for each of 60 candidates, totalling 240 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  24 tasks      | elapsed:  2.3min
    [Parallel(n_jobs=-1)]: Done 120 tasks      | elapsed: 11.2min
    [Parallel(n_jobs=-1)]: Done 240 out of 240 | elapsed: 22.4min finished


    Training Best Score:  0.9982498817867074 
    
    Training Best Params:  
     {'scale_pos_weight': 1, 'objective': 'binary:hinge', 'min_depth': 6, 'min_child_weight': 5, 'eta': 0.1} 
    
    
    Training Best Estimator:  
     XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=1, eta=0.1, gamma=0,
                  learning_rate=0.1, max_delta_step=0, max_depth=3,
                  min_child_weight=5, min_depth=6, missing=None, n_estimators=100,
                  n_jobs=1, nthread=None, objective='binary:hinge', random_state=0,
                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                  silent=None, subsample=1, verbosity=1) 
    
    
    ------------Tunning Time 1363.0429203510284 s------------ 


####LightGBM Tunning 


```python
start_time=time.time()
fit_params = { 
             "early_stopping_rounds" : 50,  ## we use early_stopping_round to stop the training of the eval metric is no longer changing (avoid overfitting)
             "eval_metric" : 'binary', 
             "eval_set" : [(X_base_test,y_base_test)],
             'eval_names': ['valid'],
             'verbose': 0}

param_test = { 
    'learning_rate' : [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4],
              'n_estimators' : [100, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000, 3000, 5000],
              #n_estimators is set to a "large value". The actual number of trees build will depend on early stopping and 5000 define only the absolute maximum
              'num_leaves': randint(6, 50), 
              'min_child_samples': randint(100, 500), 
              'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
              'subsample': uniform(loc=0.2, scale=0.8), 
              'max_depth': [-1, 1, 2, 3, 4, 5, 6, 7],
              'colsample_bytree': uniform(loc=0.4, scale=0.6),
              'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
              'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}

#number of combinations
n_iter = 200 

#intialize lgbm and lunch the search
lgbm_clf = lgb.LGBMClassifier(random_state=666, silent=False, metric='None', n_jobs=4)
grid_search = RandomizedSearchCV(
    estimator=lgbm_clf, param_distributions=param_test, 
    n_iter=n_iter,
    scoring='accuracy',
    cv=5,
    refit=True,
    random_state=666,
    verbose=True)

grid_search.fit(X_base_train, y_base_train, **fit_params)
print('Best score reached: {} with params: {} '.format(grid_search.best_score_, grid_search.best_params_))

opt_parameters =  grid_search.best_params_
#opt_parameters = {'colsample_bytree': 0.5236485492981339, 'learning_rate': 0.1, 'max_depth': -1, 'min_child_samples': 184, 'min_child_weight': 1e-05, 'n_estimators': 2000, 'num_leaves': 37, 'reg_alpha': 0.1, 'reg_lambda': 100, 'subsample': 0.6814395442651335} 
print("Timming:",time.time()-start_time)
tuned_model_scores["lgbm1"]=grid_search.best_score_




```

    Fitting 5 folds for each of 200 candidates, totalling 1000 fits


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done 1000 out of 1000 | elapsed: 566.0min finished


    Best score reached: 0.9989020769061057 with params: {'colsample_bytree': 0.5236485492981339, 'learning_rate': 0.1, 'max_depth': -1, 'min_child_samples': 184, 'min_child_weight': 1e-05, 'n_estimators': 2000, 'num_leaves': 37, 'reg_alpha': 0.1, 'reg_lambda': 100, 'subsample': 0.6814395442651335} 
    Timming: 34059.566041469574


 ***`Next`***  we try to  mix  our strategies, we run Random Search on set of hyperparamters, determine the best set among those tried. Initiate a new estimator and set its parameters with the output of random search and then we run a grid search on this estimator for an other different set of hyperpramters

$\implies$  Goal: we reduce time with random search without skipping some major parameter values by compensating for that by using a grid search over another of important sets of hyperparamters that we need to search over it exhaustively. 


```python

fit_params={"early_stopping_rounds":30, 
            "eval_metric" : 'binary_logloss', 
            "eval_set" : [(X_base_test,y_base_test)],
            'eval_names': ['valid'],
            'verbose': 100,
            'categorical_feature': 'auto'}

param_test ={'num_leaves': sp_randint(6, 50), 
             'min_child_samples': sp_randint(100, 500), 
             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             'subsample': sp_uniform(loc=0.2, scale=0.8), 
             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}




#n_estimators is set to a "large value". The actual number of trees build will depend on early stopping and 5000 define only the absolute maximum
clf = lgb.LGBMClassifier(max_depth=-1, random_state=314, silent=True, metric='None', n_jobs=4, n_estimators=5000)
gs = RandomizedSearchCV(
    estimator=clf, param_distributions=param_test, 
    n_iter=100,
    scoring='accuracy',
    cv=3,
    refit=True,
    random_state=314,
    verbose=True)

clf_sw = lgb.LGBMClassifier(**clf.get_params())
gs.fit(X_base_train, y_base_train, **fit_params)
print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))
opt_parameters = gs.best_params_
#set optimal parameters
clf_sw.set_params(**opt_parameters)

gs_sample_weight = GridSearchCV(estimator=clf_sw, 
                                param_grid={'scale_pos_weight':[1,2,6,12]},
                                scoring='accuracy',
                                cv=3,
                                refit=True,
                                verbose=True)


gs_sample_weight.fit(X_base_train, y_base_train, **fit_params)
print('Best score reached: {} with params: {} '.format(gs_sample_weight.best_score_, gs_sample_weight.best_params_))

tuned_model_scores["lgbm2"]=gs_sample_weight.best_score_

```

    Fitting 3 folds for each of 100 candidates, totalling 300 fits


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.


    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00492922
    [200]	valid's binary_logloss: 0.00416918
    [300]	valid's binary_logloss: 0.00389234
    [400]	valid's binary_logloss: 0.00373652
    [500]	valid's binary_logloss: 0.0036475
    [600]	valid's binary_logloss: 0.00359889
    [700]	valid's binary_logloss: 0.00356482
    [800]	valid's binary_logloss: 0.0035412
    [900]	valid's binary_logloss: 0.00352275
    Early stopping, best iteration is:
    [927]	valid's binary_logloss: 0.00351921
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00500393
    [200]	valid's binary_logloss: 0.0042256
    [300]	valid's binary_logloss: 0.00395023
    [400]	valid's binary_logloss: 0.00381596
    [500]	valid's binary_logloss: 0.00373379
    [600]	valid's binary_logloss: 0.00368348
    [700]	valid's binary_logloss: 0.00364413
    [800]	valid's binary_logloss: 0.0036197
    [900]	valid's binary_logloss: 0.00359164
    [1000]	valid's binary_logloss: 0.00356601
    [1100]	valid's binary_logloss: 0.00355557
    [1200]	valid's binary_logloss: 0.00354476
    [1300]	valid's binary_logloss: 0.00353346
    [1400]	valid's binary_logloss: 0.00352819
    Early stopping, best iteration is:
    [1371]	valid's binary_logloss: 0.00352781
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00492089
    [200]	valid's binary_logloss: 0.00414336
    [300]	valid's binary_logloss: 0.00387093
    [400]	valid's binary_logloss: 0.00373963
    [500]	valid's binary_logloss: 0.00364771
    [600]	valid's binary_logloss: 0.00358239
    [700]	valid's binary_logloss: 0.00353957
    [800]	valid's binary_logloss: 0.00350767
    [900]	valid's binary_logloss: 0.00347788
    [1000]	valid's binary_logloss: 0.00345701
    [1100]	valid's binary_logloss: 0.00344497
    Early stopping, best iteration is:
    [1154]	valid's binary_logloss: 0.00343696
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00513509
    [200]	valid's binary_logloss: 0.00434819
    [300]	valid's binary_logloss: 0.00407078
    [400]	valid's binary_logloss: 0.00394486
    [500]	valid's binary_logloss: 0.00387278
    [600]	valid's binary_logloss: 0.00384099
    Early stopping, best iteration is:
    [575]	valid's binary_logloss: 0.00384085
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00511832
    [200]	valid's binary_logloss: 0.00438562
    [300]	valid's binary_logloss: 0.00414151
    [400]	valid's binary_logloss: 0.00401487
    [500]	valid's binary_logloss: 0.00395276
    Early stopping, best iteration is:
    [537]	valid's binary_logloss: 0.00393386
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00506638
    [200]	valid's binary_logloss: 0.00432311
    [300]	valid's binary_logloss: 0.00405851
    [400]	valid's binary_logloss: 0.00393253
    [500]	valid's binary_logloss: 0.00385794
    [600]	valid's binary_logloss: 0.00381709
    Early stopping, best iteration is:
    [621]	valid's binary_logloss: 0.0038123
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [2]	valid's binary_logloss: 0.2404
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [1]	valid's binary_logloss: 0.240396
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [2]	valid's binary_logloss: 0.2404
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [46]	valid's binary_logloss: 0.0108862
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [47]	valid's binary_logloss: 0.0107758
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [50]	valid's binary_logloss: 0.0107579
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.0064059
    [200]	valid's binary_logloss: 0.00629177
    [300]	valid's binary_logloss: 0.00629172
    [400]	valid's binary_logloss: 0.00629172
    [500]	valid's binary_logloss: 0.00629172
    Early stopping, best iteration is:
    [473]	valid's binary_logloss: 0.00629172
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00647317
    [200]	valid's binary_logloss: 0.00638916
    [300]	valid's binary_logloss: 0.00638915
    [400]	valid's binary_logloss: 0.00638915
    [500]	valid's binary_logloss: 0.00638915
    Early stopping, best iteration is:
    [481]	valid's binary_logloss: 0.00638915
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00627685
    [200]	valid's binary_logloss: 0.00619821
    [300]	valid's binary_logloss: 0.00619821
    [400]	valid's binary_logloss: 0.00619821
    [500]	valid's binary_logloss: 0.00619821
    Early stopping, best iteration is:
    [476]	valid's binary_logloss: 0.00619821
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00453593
    [200]	valid's binary_logloss: 0.00401934
    [300]	valid's binary_logloss: 0.00382741
    [400]	valid's binary_logloss: 0.00373208
    [500]	valid's binary_logloss: 0.00368421
    [600]	valid's binary_logloss: 0.00364851
    Early stopping, best iteration is:
    [632]	valid's binary_logloss: 0.00363638
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00450184
    [200]	valid's binary_logloss: 0.00400837
    [300]	valid's binary_logloss: 0.00383071
    [400]	valid's binary_logloss: 0.00374547
    [500]	valid's binary_logloss: 0.00370518
    [600]	valid's binary_logloss: 0.00367122
    [700]	valid's binary_logloss: 0.00363289
    Early stopping, best iteration is:
    [687]	valid's binary_logloss: 0.00362895
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00447479
    [200]	valid's binary_logloss: 0.00397424
    [300]	valid's binary_logloss: 0.00381377
    [400]	valid's binary_logloss: 0.00370512
    [500]	valid's binary_logloss: 0.00363981
    [600]	valid's binary_logloss: 0.00360127
    [700]	valid's binary_logloss: 0.00357595
    Early stopping, best iteration is:
    [710]	valid's binary_logloss: 0.00357161
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00391446
    [200]	valid's binary_logloss: 0.00363545
    [300]	valid's binary_logloss: 0.00359419
    Early stopping, best iteration is:
    [283]	valid's binary_logloss: 0.00359024
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00394976
    [200]	valid's binary_logloss: 0.0036601
    [300]	valid's binary_logloss: 0.00360067
    Early stopping, best iteration is:
    [361]	valid's binary_logloss: 0.0035875
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00397763
    [200]	valid's binary_logloss: 0.00367319
    [300]	valid's binary_logloss: 0.00363314
    Early stopping, best iteration is:
    [348]	valid's binary_logloss: 0.00362005
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00408334
    [200]	valid's binary_logloss: 0.00362376
    [300]	valid's binary_logloss: 0.00353805
    Early stopping, best iteration is:
    [323]	valid's binary_logloss: 0.00353288
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00415805
    [200]	valid's binary_logloss: 0.00365389
    [300]	valid's binary_logloss: 0.00352126
    Early stopping, best iteration is:
    [360]	valid's binary_logloss: 0.00349311
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00410225
    [200]	valid's binary_logloss: 0.00360336
    [300]	valid's binary_logloss: 0.003457
    [400]	valid's binary_logloss: 0.00341788
    Early stopping, best iteration is:
    [417]	valid's binary_logloss: 0.00341707
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00440423
    [200]	valid's binary_logloss: 0.00387034
    [300]	valid's binary_logloss: 0.00371033
    [400]	valid's binary_logloss: 0.00360165
    [500]	valid's binary_logloss: 0.00357449
    Early stopping, best iteration is:
    [533]	valid's binary_logloss: 0.00356927
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00438047
    [200]	valid's binary_logloss: 0.00394023
    [300]	valid's binary_logloss: 0.00374898
    [400]	valid's binary_logloss: 0.00366756
    [500]	valid's binary_logloss: 0.00360195
    [600]	valid's binary_logloss: 0.00355017
    [700]	valid's binary_logloss: 0.00352466
    Early stopping, best iteration is:
    [709]	valid's binary_logloss: 0.00352377
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.0043248
    [200]	valid's binary_logloss: 0.00381307
    [300]	valid's binary_logloss: 0.00362952
    [400]	valid's binary_logloss: 0.00355457
    [500]	valid's binary_logloss: 0.00349202
    [600]	valid's binary_logloss: 0.00344632
    Early stopping, best iteration is:
    [614]	valid's binary_logloss: 0.00344249
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00409948
    [200]	valid's binary_logloss: 0.00368414
    [300]	valid's binary_logloss: 0.00359132
    Early stopping, best iteration is:
    [341]	valid's binary_logloss: 0.00356754
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00417302
    [200]	valid's binary_logloss: 0.00373666
    [300]	valid's binary_logloss: 0.00361521
    [400]	valid's binary_logloss: 0.0035674
    Early stopping, best iteration is:
    [434]	valid's binary_logloss: 0.00355531
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00416082
    [200]	valid's binary_logloss: 0.00375837
    [300]	valid's binary_logloss: 0.00361116
    [400]	valid's binary_logloss: 0.00356991
    [500]	valid's binary_logloss: 0.00356789
    Early stopping, best iteration is:
    [476]	valid's binary_logloss: 0.00356396
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00459383
    [200]	valid's binary_logloss: 0.00402309
    [300]	valid's binary_logloss: 0.00384075
    [400]	valid's binary_logloss: 0.00378274
    Early stopping, best iteration is:
    [377]	valid's binary_logloss: 0.0037824
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00462083
    [200]	valid's binary_logloss: 0.00406038
    [300]	valid's binary_logloss: 0.00387349
    [400]	valid's binary_logloss: 0.00380793
    Early stopping, best iteration is:
    [406]	valid's binary_logloss: 0.00380727
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.0045858
    [200]	valid's binary_logloss: 0.00401808
    [300]	valid's binary_logloss: 0.0038236
    [400]	valid's binary_logloss: 0.00374116
    Early stopping, best iteration is:
    [389]	valid's binary_logloss: 0.00374116
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00447298
    [200]	valid's binary_logloss: 0.00396779
    [300]	valid's binary_logloss: 0.00381902
    Early stopping, best iteration is:
    [350]	valid's binary_logloss: 0.00378112
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00441193
    [200]	valid's binary_logloss: 0.00390871
    [300]	valid's binary_logloss: 0.00374638
    Early stopping, best iteration is:
    [367]	valid's binary_logloss: 0.00369758
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00437387
    [200]	valid's binary_logloss: 0.00384487
    [300]	valid's binary_logloss: 0.0036759
    [400]	valid's binary_logloss: 0.00359949
    Early stopping, best iteration is:
    [404]	valid's binary_logloss: 0.0035979
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.0042557
    [200]	valid's binary_logloss: 0.00379609
    [300]	valid's binary_logloss: 0.00362532
    [400]	valid's binary_logloss: 0.00355769
    [500]	valid's binary_logloss: 0.00353061
    Early stopping, best iteration is:
    [478]	valid's binary_logloss: 0.0035276
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00437316
    [200]	valid's binary_logloss: 0.00389404
    [300]	valid's binary_logloss: 0.00374712
    [400]	valid's binary_logloss: 0.00368525
    [500]	valid's binary_logloss: 0.00364817
    [600]	valid's binary_logloss: 0.00363177
    [700]	valid's binary_logloss: 0.00361362
    Early stopping, best iteration is:
    [741]	valid's binary_logloss: 0.00360712
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00433772
    [200]	valid's binary_logloss: 0.00381183
    [300]	valid's binary_logloss: 0.0036401
    [400]	valid's binary_logloss: 0.00356805
    [500]	valid's binary_logloss: 0.00353181
    [600]	valid's binary_logloss: 0.00351139
    [700]	valid's binary_logloss: 0.00349638
    Early stopping, best iteration is:
    [702]	valid's binary_logloss: 0.00349584
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [1]	valid's binary_logloss: 0.2404
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [1]	valid's binary_logloss: 0.240396
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [5]	valid's binary_logloss: 0.2404
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [51]	valid's binary_logloss: 0.0109239
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [50]	valid's binary_logloss: 0.0108093
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [53]	valid's binary_logloss: 0.0106474
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00519714
    [200]	valid's binary_logloss: 0.00440148
    [300]	valid's binary_logloss: 0.00410072
    [400]	valid's binary_logloss: 0.0039527
    [500]	valid's binary_logloss: 0.00385154
    [600]	valid's binary_logloss: 0.00378217
    [700]	valid's binary_logloss: 0.00373669
    [800]	valid's binary_logloss: 0.00369597
    [900]	valid's binary_logloss: 0.0036666
    [1000]	valid's binary_logloss: 0.00364226
    [1100]	valid's binary_logloss: 0.00362734
    [1200]	valid's binary_logloss: 0.00361937
    Early stopping, best iteration is:
    [1175]	valid's binary_logloss: 0.00361763
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00521863
    [200]	valid's binary_logloss: 0.00443503
    [300]	valid's binary_logloss: 0.00414818
    [400]	valid's binary_logloss: 0.00398367
    [500]	valid's binary_logloss: 0.00388583
    [600]	valid's binary_logloss: 0.00381084
    [700]	valid's binary_logloss: 0.00376104
    [800]	valid's binary_logloss: 0.00372961
    [900]	valid's binary_logloss: 0.00369845
    [1000]	valid's binary_logloss: 0.00367455
    [1100]	valid's binary_logloss: 0.00366532
    [1200]	valid's binary_logloss: 0.00365417
    [1300]	valid's binary_logloss: 0.00364203
    [1400]	valid's binary_logloss: 0.00363015
    Early stopping, best iteration is:
    [1382]	valid's binary_logloss: 0.00362921
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.0051489
    [200]	valid's binary_logloss: 0.0043564
    [300]	valid's binary_logloss: 0.0040767
    [400]	valid's binary_logloss: 0.00392487
    [500]	valid's binary_logloss: 0.00382822
    [600]	valid's binary_logloss: 0.00375932
    [700]	valid's binary_logloss: 0.00370348
    [800]	valid's binary_logloss: 0.00366011
    [900]	valid's binary_logloss: 0.00363109
    [1000]	valid's binary_logloss: 0.00360562
    [1100]	valid's binary_logloss: 0.00358683
    [1200]	valid's binary_logloss: 0.00357214
    [1300]	valid's binary_logloss: 0.00356398
    [1400]	valid's binary_logloss: 0.00355673
    [1500]	valid's binary_logloss: 0.00355059
    Early stopping, best iteration is:
    [1511]	valid's binary_logloss: 0.00354988
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00587815
    [200]	valid's binary_logloss: 0.00529031
    [300]	valid's binary_logloss: 0.00505983
    [400]	valid's binary_logloss: 0.00492749
    [500]	valid's binary_logloss: 0.00484012
    [600]	valid's binary_logloss: 0.00478858
    [700]	valid's binary_logloss: 0.00474013
    [800]	valid's binary_logloss: 0.00470306
    [900]	valid's binary_logloss: 0.00467232
    [1000]	valid's binary_logloss: 0.00464704
    [1100]	valid's binary_logloss: 0.00462602
    [1200]	valid's binary_logloss: 0.004601
    [1300]	valid's binary_logloss: 0.00458169
    [1400]	valid's binary_logloss: 0.00456926
    [1500]	valid's binary_logloss: 0.00455031
    [1600]	valid's binary_logloss: 0.00453559
    [1700]	valid's binary_logloss: 0.00452862
    [1800]	valid's binary_logloss: 0.00452318
    [1900]	valid's binary_logloss: 0.00451057
    [2000]	valid's binary_logloss: 0.00449981
    [2100]	valid's binary_logloss: 0.00449007
    [2200]	valid's binary_logloss: 0.00447995
    [2300]	valid's binary_logloss: 0.00447141
    [2400]	valid's binary_logloss: 0.00446703
    Early stopping, best iteration is:
    [2384]	valid's binary_logloss: 0.00446591
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00586339
    [200]	valid's binary_logloss: 0.00525361
    [300]	valid's binary_logloss: 0.00501883
    [400]	valid's binary_logloss: 0.00488129
    [500]	valid's binary_logloss: 0.00478893
    [600]	valid's binary_logloss: 0.00472241
    [700]	valid's binary_logloss: 0.00467028
    [800]	valid's binary_logloss: 0.00463331
    [900]	valid's binary_logloss: 0.00460358
    [1000]	valid's binary_logloss: 0.00457316
    [1100]	valid's binary_logloss: 0.00454485
    [1200]	valid's binary_logloss: 0.00452637
    [1300]	valid's binary_logloss: 0.00451514
    [1400]	valid's binary_logloss: 0.00450241
    [1500]	valid's binary_logloss: 0.00449176
    [1600]	valid's binary_logloss: 0.00447783
    [1700]	valid's binary_logloss: 0.00447091
    [1800]	valid's binary_logloss: 0.00445815
    [1900]	valid's binary_logloss: 0.00444533
    [2000]	valid's binary_logloss: 0.0044399
    Early stopping, best iteration is:
    [2065]	valid's binary_logloss: 0.00443155
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00583406
    [200]	valid's binary_logloss: 0.0052332
    [300]	valid's binary_logloss: 0.00499813
    [400]	valid's binary_logloss: 0.00486532
    [500]	valid's binary_logloss: 0.00477273
    [600]	valid's binary_logloss: 0.00470261
    [700]	valid's binary_logloss: 0.00465264
    [800]	valid's binary_logloss: 0.00460506
    [900]	valid's binary_logloss: 0.00457055
    [1000]	valid's binary_logloss: 0.00454388
    [1100]	valid's binary_logloss: 0.00452283
    [1200]	valid's binary_logloss: 0.00449701
    [1300]	valid's binary_logloss: 0.00447916
    [1400]	valid's binary_logloss: 0.00446736
    [1500]	valid's binary_logloss: 0.00445054
    [1600]	valid's binary_logloss: 0.00443815
    [1700]	valid's binary_logloss: 0.00442591
    [1800]	valid's binary_logloss: 0.00441497
    [1900]	valid's binary_logloss: 0.00440452
    [2000]	valid's binary_logloss: 0.00439498
    [2100]	valid's binary_logloss: 0.00438522
    [2200]	valid's binary_logloss: 0.00437698
    [2300]	valid's binary_logloss: 0.0043691
    [2400]	valid's binary_logloss: 0.00436063
    Early stopping, best iteration is:
    [2405]	valid's binary_logloss: 0.00435903
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.0047329
    [200]	valid's binary_logloss: 0.00410277
    [300]	valid's binary_logloss: 0.00386517
    [400]	valid's binary_logloss: 0.00373879
    [500]	valid's binary_logloss: 0.00364445
    [600]	valid's binary_logloss: 0.00358124
    [700]	valid's binary_logloss: 0.00355482
    Early stopping, best iteration is:
    [694]	valid's binary_logloss: 0.00355309
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00477766
    [200]	valid's binary_logloss: 0.00411054
    [300]	valid's binary_logloss: 0.00387906
    [400]	valid's binary_logloss: 0.00377109
    [500]	valid's binary_logloss: 0.00369729
    [600]	valid's binary_logloss: 0.0036537
    [700]	valid's binary_logloss: 0.00361358
    [800]	valid's binary_logloss: 0.00358683
    [900]	valid's binary_logloss: 0.00357419
    [1000]	valid's binary_logloss: 0.00356267
    Early stopping, best iteration is:
    [1054]	valid's binary_logloss: 0.00355658
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00472786
    [200]	valid's binary_logloss: 0.00406712
    [300]	valid's binary_logloss: 0.00383258
    [400]	valid's binary_logloss: 0.00370856
    [500]	valid's binary_logloss: 0.00362044
    [600]	valid's binary_logloss: 0.00356635
    [700]	valid's binary_logloss: 0.00352936
    [800]	valid's binary_logloss: 0.00350581
    Early stopping, best iteration is:
    [839]	valid's binary_logloss: 0.00349291
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00618729
    [200]	valid's binary_logloss: 0.00577805
    Early stopping, best iteration is:
    [215]	valid's binary_logloss: 0.00576496
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00613405
    [200]	valid's binary_logloss: 0.00573774
    Early stopping, best iteration is:
    [229]	valid's binary_logloss: 0.00570825
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00613511
    [200]	valid's binary_logloss: 0.00572267
    Early stopping, best iteration is:
    [222]	valid's binary_logloss: 0.00570736
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00377757
    Early stopping, best iteration is:
    [159]	valid's binary_logloss: 0.00366136
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00391073
    [200]	valid's binary_logloss: 0.00371519
    Early stopping, best iteration is:
    [201]	valid's binary_logloss: 0.00371443
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00379271
    [200]	valid's binary_logloss: 0.00364004
    Early stopping, best iteration is:
    [193]	valid's binary_logloss: 0.00362885
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00639525
    Early stopping, best iteration is:
    [161]	valid's binary_logloss: 0.00634153
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00644113
    [200]	valid's binary_logloss: 0.00638492
    [300]	valid's binary_logloss: 0.00638492
    [400]	valid's binary_logloss: 0.00638492
    [500]	valid's binary_logloss: 0.00638492
    Early stopping, best iteration is:
    [496]	valid's binary_logloss: 0.00638492
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00643977
    [200]	valid's binary_logloss: 0.00637543
    [300]	valid's binary_logloss: 0.00637543
    [400]	valid's binary_logloss: 0.00637543
    [500]	valid's binary_logloss: 0.00637543
    [600]	valid's binary_logloss: 0.00637543
    Early stopping, best iteration is:
    [577]	valid's binary_logloss: 0.00637543
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00474991
    [200]	valid's binary_logloss: 0.00412938
    [300]	valid's binary_logloss: 0.00390989
    [400]	valid's binary_logloss: 0.00378192
    [500]	valid's binary_logloss: 0.00371078
    [600]	valid's binary_logloss: 0.00366593
    [700]	valid's binary_logloss: 0.00362492
    [800]	valid's binary_logloss: 0.00361407
    Early stopping, best iteration is:
    [825]	valid's binary_logloss: 0.00361038
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00469063
    [200]	valid's binary_logloss: 0.0041213
    [300]	valid's binary_logloss: 0.00391204
    [400]	valid's binary_logloss: 0.00379511
    [500]	valid's binary_logloss: 0.00371693
    [600]	valid's binary_logloss: 0.00368034
    [700]	valid's binary_logloss: 0.00363527
    [800]	valid's binary_logloss: 0.00360011
    [900]	valid's binary_logloss: 0.00358073
    [1000]	valid's binary_logloss: 0.00356452
    Early stopping, best iteration is:
    [983]	valid's binary_logloss: 0.00356302
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00469653
    [200]	valid's binary_logloss: 0.00411208
    [300]	valid's binary_logloss: 0.00391521
    [400]	valid's binary_logloss: 0.00378664
    [500]	valid's binary_logloss: 0.00371976
    [600]	valid's binary_logloss: 0.00366544
    [700]	valid's binary_logloss: 0.00362601
    [800]	valid's binary_logloss: 0.00360082
    [900]	valid's binary_logloss: 0.00356134
    Early stopping, best iteration is:
    [903]	valid's binary_logloss: 0.00355863
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00679943
    [200]	valid's binary_logloss: 0.00659914
    Early stopping, best iteration is:
    [172]	valid's binary_logloss: 0.00659914
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00679957
    [200]	valid's binary_logloss: 0.0066305
    [300]	valid's binary_logloss: 0.0066305
    [400]	valid's binary_logloss: 0.0066305
    Early stopping, best iteration is:
    [400]	valid's binary_logloss: 0.0066305
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00669157
    [200]	valid's binary_logloss: 0.00648622
    [300]	valid's binary_logloss: 0.00648616
    [400]	valid's binary_logloss: 0.00648616
    Early stopping, best iteration is:
    [393]	valid's binary_logloss: 0.00648616
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00506739
    [200]	valid's binary_logloss: 0.00438391
    [300]	valid's binary_logloss: 0.00412426
    [400]	valid's binary_logloss: 0.00397701
    [500]	valid's binary_logloss: 0.00387947
    [600]	valid's binary_logloss: 0.00381198
    [700]	valid's binary_logloss: 0.00376147
    [800]	valid's binary_logloss: 0.00372433
    [900]	valid's binary_logloss: 0.00369382
    [1000]	valid's binary_logloss: 0.00367159
    [1100]	valid's binary_logloss: 0.00364826
    Early stopping, best iteration is:
    [1125]	valid's binary_logloss: 0.00364228
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00505805
    [200]	valid's binary_logloss: 0.00435727
    [300]	valid's binary_logloss: 0.00411251
    [400]	valid's binary_logloss: 0.003981
    [500]	valid's binary_logloss: 0.00387716
    [600]	valid's binary_logloss: 0.00382336
    [700]	valid's binary_logloss: 0.00378362
    [800]	valid's binary_logloss: 0.00375058
    [900]	valid's binary_logloss: 0.00372159
    [1000]	valid's binary_logloss: 0.00369747
    [1100]	valid's binary_logloss: 0.00368056
    Early stopping, best iteration is:
    [1150]	valid's binary_logloss: 0.00367177
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00498376
    [200]	valid's binary_logloss: 0.00431573
    [300]	valid's binary_logloss: 0.00406635
    [400]	valid's binary_logloss: 0.0039091
    [500]	valid's binary_logloss: 0.00381275
    [600]	valid's binary_logloss: 0.00373713
    [700]	valid's binary_logloss: 0.00368048
    [800]	valid's binary_logloss: 0.00363731
    [900]	valid's binary_logloss: 0.00360461
    [1000]	valid's binary_logloss: 0.00356675
    [1100]	valid's binary_logloss: 0.00353984
    [1200]	valid's binary_logloss: 0.00353009
    [1300]	valid's binary_logloss: 0.00351785
    Early stopping, best iteration is:
    [1340]	valid's binary_logloss: 0.00350992
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00432019
    [200]	valid's binary_logloss: 0.0038032
    [300]	valid's binary_logloss: 0.00365598
    [400]	valid's binary_logloss: 0.00363016
    Early stopping, best iteration is:
    [463]	valid's binary_logloss: 0.00361531
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00427641
    [200]	valid's binary_logloss: 0.003767
    [300]	valid's binary_logloss: 0.00363287
    [400]	valid's binary_logloss: 0.00357133
    [500]	valid's binary_logloss: 0.0035426
    [600]	valid's binary_logloss: 0.0035361
    Early stopping, best iteration is:
    [587]	valid's binary_logloss: 0.00353215
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00426523
    [200]	valid's binary_logloss: 0.00375159
    [300]	valid's binary_logloss: 0.00359465
    [400]	valid's binary_logloss: 0.00353338
    [500]	valid's binary_logloss: 0.00350532
    Early stopping, best iteration is:
    [494]	valid's binary_logloss: 0.00350411
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [1]	valid's binary_logloss: 0.2404
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [2]	valid's binary_logloss: 0.240396
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [1]	valid's binary_logloss: 0.2404
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00428927
    [200]	valid's binary_logloss: 0.00384883
    [300]	valid's binary_logloss: 0.00373074
    Early stopping, best iteration is:
    [367]	valid's binary_logloss: 0.00370081
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00428659
    [200]	valid's binary_logloss: 0.00387012
    [300]	valid's binary_logloss: 0.00374681
    Early stopping, best iteration is:
    [341]	valid's binary_logloss: 0.00371698
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00423883
    [200]	valid's binary_logloss: 0.00382541
    [300]	valid's binary_logloss: 0.00368769
    Early stopping, best iteration is:
    [335]	valid's binary_logloss: 0.00365944
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00483978
    [200]	valid's binary_logloss: 0.00422469
    [300]	valid's binary_logloss: 0.00399403
    [400]	valid's binary_logloss: 0.00388467
    [500]	valid's binary_logloss: 0.00380854
    [600]	valid's binary_logloss: 0.00375717
    [700]	valid's binary_logloss: 0.0037256
    [800]	valid's binary_logloss: 0.00369903
    [900]	valid's binary_logloss: 0.0036879
    [1000]	valid's binary_logloss: 0.00367721
    Early stopping, best iteration is:
    [1009]	valid's binary_logloss: 0.00367682
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00486673
    [200]	valid's binary_logloss: 0.00421135
    [300]	valid's binary_logloss: 0.00398991
    [400]	valid's binary_logloss: 0.00387808
    [500]	valid's binary_logloss: 0.00381517
    [600]	valid's binary_logloss: 0.00376354
    [700]	valid's binary_logloss: 0.0037321
    [800]	valid's binary_logloss: 0.00370406
    Early stopping, best iteration is:
    [837]	valid's binary_logloss: 0.00370051
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00477895
    [200]	valid's binary_logloss: 0.00413433
    [300]	valid's binary_logloss: 0.00391677
    [400]	valid's binary_logloss: 0.0037929
    [500]	valid's binary_logloss: 0.00372661
    [600]	valid's binary_logloss: 0.00368962
    [700]	valid's binary_logloss: 0.00366461
    [800]	valid's binary_logloss: 0.00363976
    [900]	valid's binary_logloss: 0.00362856
    [1000]	valid's binary_logloss: 0.00361695
    [1100]	valid's binary_logloss: 0.0036129
    [1200]	valid's binary_logloss: 0.0036129
    [1300]	valid's binary_logloss: 0.0036129
    [1400]	valid's binary_logloss: 0.0036129
    Early stopping, best iteration is:
    [1399]	valid's binary_logloss: 0.0036129
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00514157
    [200]	valid's binary_logloss: 0.00445358
    [300]	valid's binary_logloss: 0.00420267
    [400]	valid's binary_logloss: 0.00407648
    [500]	valid's binary_logloss: 0.00399928
    [600]	valid's binary_logloss: 0.00394136
    [700]	valid's binary_logloss: 0.00390328
    [800]	valid's binary_logloss: 0.00386409
    [900]	valid's binary_logloss: 0.00384992
    Early stopping, best iteration is:
    [946]	valid's binary_logloss: 0.0038371
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00517646
    [200]	valid's binary_logloss: 0.00450754
    [300]	valid's binary_logloss: 0.00424404
    [400]	valid's binary_logloss: 0.00411547
    [500]	valid's binary_logloss: 0.00402219
    [600]	valid's binary_logloss: 0.00396194
    [700]	valid's binary_logloss: 0.00392084
    Early stopping, best iteration is:
    [749]	valid's binary_logloss: 0.00390131
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00511417
    [200]	valid's binary_logloss: 0.00443881
    [300]	valid's binary_logloss: 0.00420594
    [400]	valid's binary_logloss: 0.00407121
    [500]	valid's binary_logloss: 0.00399563
    [600]	valid's binary_logloss: 0.00391708
    [700]	valid's binary_logloss: 0.00388041
    [800]	valid's binary_logloss: 0.00383846
    [900]	valid's binary_logloss: 0.00380234
    [1000]	valid's binary_logloss: 0.00378921
    Early stopping, best iteration is:
    [1006]	valid's binary_logloss: 0.00378841
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00596173
    [200]	valid's binary_logloss: 0.00533998
    [300]	valid's binary_logloss: 0.00526571
    [400]	valid's binary_logloss: 0.00526566
    [500]	valid's binary_logloss: 0.00526565
    [600]	valid's binary_logloss: 0.00526565
    [700]	valid's binary_logloss: 0.00526565
    [800]	valid's binary_logloss: 0.00526565
    [900]	valid's binary_logloss: 0.00526565
    [1000]	valid's binary_logloss: 0.00526565
    Early stopping, best iteration is:
    [1005]	valid's binary_logloss: 0.00526565
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00599939
    [200]	valid's binary_logloss: 0.00540319
    [300]	valid's binary_logloss: 0.00532239
    [400]	valid's binary_logloss: 0.00532235
    [500]	valid's binary_logloss: 0.00532235
    [600]	valid's binary_logloss: 0.00532235
    [700]	valid's binary_logloss: 0.00532235
    [800]	valid's binary_logloss: 0.00532235
    Early stopping, best iteration is:
    [772]	valid's binary_logloss: 0.00532235
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00590976
    [200]	valid's binary_logloss: 0.00533926
    [300]	valid's binary_logloss: 0.00526165
    [400]	valid's binary_logloss: 0.0052615
    [500]	valid's binary_logloss: 0.0052615
    [600]	valid's binary_logloss: 0.0052615
    [700]	valid's binary_logloss: 0.0052615
    [800]	valid's binary_logloss: 0.0052615
    [900]	valid's binary_logloss: 0.0052615
    Early stopping, best iteration is:
    [908]	valid's binary_logloss: 0.0052615
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00633409
    [200]	valid's binary_logloss: 0.00628103
    [300]	valid's binary_logloss: 0.00628103
    Early stopping, best iteration is:
    [315]	valid's binary_logloss: 0.00628103
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00642143
    Early stopping, best iteration is:
    [145]	valid's binary_logloss: 0.00636023
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.0062597
    [200]	valid's binary_logloss: 0.00618069
    [300]	valid's binary_logloss: 0.00618069
    [400]	valid's binary_logloss: 0.00618069
    Early stopping, best iteration is:
    [453]	valid's binary_logloss: 0.00618069
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00434333
    [200]	valid's binary_logloss: 0.00383527
    [300]	valid's binary_logloss: 0.00367579
    [400]	valid's binary_logloss: 0.00362416
    [500]	valid's binary_logloss: 0.00360489
    Early stopping, best iteration is:
    [555]	valid's binary_logloss: 0.00359196
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00432406
    [200]	valid's binary_logloss: 0.00380972
    [300]	valid's binary_logloss: 0.00364293
    [400]	valid's binary_logloss: 0.00358226
    [500]	valid's binary_logloss: 0.00354252
    Early stopping, best iteration is:
    [526]	valid's binary_logloss: 0.00352748
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00434767
    [200]	valid's binary_logloss: 0.00382642
    [300]	valid's binary_logloss: 0.00363897
    [400]	valid's binary_logloss: 0.00357223
    [500]	valid's binary_logloss: 0.00353458
    [600]	valid's binary_logloss: 0.00351737
    Early stopping, best iteration is:
    [625]	valid's binary_logloss: 0.00351724
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00435132
    [200]	valid's binary_logloss: 0.00397049
    Early stopping, best iteration is:
    [194]	valid's binary_logloss: 0.00396989
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00440197
    [200]	valid's binary_logloss: 0.00401781
    Early stopping, best iteration is:
    [196]	valid's binary_logloss: 0.00401781
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00435413
    [200]	valid's binary_logloss: 0.00397382
    Early stopping, best iteration is:
    [215]	valid's binary_logloss: 0.00396466
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [1]	valid's binary_logloss: 0.2404
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [3]	valid's binary_logloss: 0.240396
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [5]	valid's binary_logloss: 0.2404
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00619778
    [200]	valid's binary_logloss: 0.00580769
    Early stopping, best iteration is:
    [231]	valid's binary_logloss: 0.00577306
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00613824
    [200]	valid's binary_logloss: 0.00578021
    Early stopping, best iteration is:
    [228]	valid's binary_logloss: 0.00574949
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00614933
    [200]	valid's binary_logloss: 0.00574922
    [300]	valid's binary_logloss: 0.00573932
    [400]	valid's binary_logloss: 0.00573931
    [500]	valid's binary_logloss: 0.00573931
    Early stopping, best iteration is:
    [493]	valid's binary_logloss: 0.00573931
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00488645
    [200]	valid's binary_logloss: 0.00424077
    [300]	valid's binary_logloss: 0.00402524
    [400]	valid's binary_logloss: 0.00391424
    [500]	valid's binary_logloss: 0.00383536
    [600]	valid's binary_logloss: 0.00379143
    [700]	valid's binary_logloss: 0.00376096
    [800]	valid's binary_logloss: 0.00374612
    Early stopping, best iteration is:
    [789]	valid's binary_logloss: 0.00374574
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00495419
    [200]	valid's binary_logloss: 0.00430801
    [300]	valid's binary_logloss: 0.0040872
    [400]	valid's binary_logloss: 0.00398443
    [500]	valid's binary_logloss: 0.00391752
    [600]	valid's binary_logloss: 0.00387499
    Early stopping, best iteration is:
    [656]	valid's binary_logloss: 0.00385187
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00489218
    [200]	valid's binary_logloss: 0.00425149
    [300]	valid's binary_logloss: 0.0040396
    [400]	valid's binary_logloss: 0.00392453
    [500]	valid's binary_logloss: 0.00385675
    [600]	valid's binary_logloss: 0.00381629
    Early stopping, best iteration is:
    [611]	valid's binary_logloss: 0.00381373
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00442414
    [200]	valid's binary_logloss: 0.003879
    [300]	valid's binary_logloss: 0.00369193
    [400]	valid's binary_logloss: 0.00359979
    [500]	valid's binary_logloss: 0.00356384
    [600]	valid's binary_logloss: 0.00354238
    Early stopping, best iteration is:
    [646]	valid's binary_logloss: 0.00353671
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00449045
    [200]	valid's binary_logloss: 0.0039415
    [300]	valid's binary_logloss: 0.00376498
    [400]	valid's binary_logloss: 0.00367811
    [500]	valid's binary_logloss: 0.00363306
    [600]	valid's binary_logloss: 0.00360536
    [700]	valid's binary_logloss: 0.00359053
    Early stopping, best iteration is:
    [703]	valid's binary_logloss: 0.00358963
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00444237
    [200]	valid's binary_logloss: 0.00388285
    [300]	valid's binary_logloss: 0.00366672
    [400]	valid's binary_logloss: 0.00357046
    [500]	valid's binary_logloss: 0.00352539
    [600]	valid's binary_logloss: 0.00350793
    Early stopping, best iteration is:
    [625]	valid's binary_logloss: 0.00349921
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [2]	valid's binary_logloss: 0.2404
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [2]	valid's binary_logloss: 0.240396
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [1]	valid's binary_logloss: 0.2404
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [49]	valid's binary_logloss: 0.0106706
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [45]	valid's binary_logloss: 0.0108514
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [50]	valid's binary_logloss: 0.0107874
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.0046906
    [200]	valid's binary_logloss: 0.00398338
    [300]	valid's binary_logloss: 0.00373067
    [400]	valid's binary_logloss: 0.00360811
    [500]	valid's binary_logloss: 0.00355037
    [600]	valid's binary_logloss: 0.00352314
    [700]	valid's binary_logloss: 0.00351271
    Early stopping, best iteration is:
    [761]	valid's binary_logloss: 0.00350519
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00476168
    [200]	valid's binary_logloss: 0.0040459
    [300]	valid's binary_logloss: 0.00378639
    [400]	valid's binary_logloss: 0.00366661
    [500]	valid's binary_logloss: 0.00359039
    [600]	valid's binary_logloss: 0.00353594
    [700]	valid's binary_logloss: 0.00351236
    [800]	valid's binary_logloss: 0.0035003
    Early stopping, best iteration is:
    [828]	valid's binary_logloss: 0.00349384
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00468156
    [200]	valid's binary_logloss: 0.00396702
    [300]	valid's binary_logloss: 0.00372063
    [400]	valid's binary_logloss: 0.00359559
    [500]	valid's binary_logloss: 0.00350489
    [600]	valid's binary_logloss: 0.00345404
    [700]	valid's binary_logloss: 0.003424
    Early stopping, best iteration is:
    [761]	valid's binary_logloss: 0.00340932
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00454198
    [200]	valid's binary_logloss: 0.0041269
    Early stopping, best iteration is:
    [230]	valid's binary_logloss: 0.0040874
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00444283
    [200]	valid's binary_logloss: 0.004067
    Early stopping, best iteration is:
    [250]	valid's binary_logloss: 0.00399986
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00442077
    [200]	valid's binary_logloss: 0.00403578
    Early stopping, best iteration is:
    [230]	valid's binary_logloss: 0.00399469
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [2]	valid's binary_logloss: 0.2404
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [2]	valid's binary_logloss: 0.240396
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [1]	valid's binary_logloss: 0.2404
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00678274
    [200]	valid's binary_logloss: 0.00660627
    [300]	valid's binary_logloss: 0.00660626
    [400]	valid's binary_logloss: 0.00660626
    [500]	valid's binary_logloss: 0.00660626
    [600]	valid's binary_logloss: 0.00660626
    [700]	valid's binary_logloss: 0.00660626
    Early stopping, best iteration is:
    [684]	valid's binary_logloss: 0.00660626
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00680462
    [200]	valid's binary_logloss: 0.00662476
    [300]	valid's binary_logloss: 0.00662475
    [400]	valid's binary_logloss: 0.00662475
    [500]	valid's binary_logloss: 0.00662475
    [600]	valid's binary_logloss: 0.00662475
    Early stopping, best iteration is:
    [573]	valid's binary_logloss: 0.00662475
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00671877
    [200]	valid's binary_logloss: 0.00651569
    [300]	valid's binary_logloss: 0.00651567
    [400]	valid's binary_logloss: 0.00651567
    Early stopping, best iteration is:
    [468]	valid's binary_logloss: 0.00651567
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00384623
    [200]	valid's binary_logloss: 0.00358953
    Early stopping, best iteration is:
    [263]	valid's binary_logloss: 0.00354981
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00393734
    [200]	valid's binary_logloss: 0.00367766
    [300]	valid's binary_logloss: 0.00361411
    Early stopping, best iteration is:
    [336]	valid's binary_logloss: 0.00360764
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00392784
    [200]	valid's binary_logloss: 0.00365729
    [300]	valid's binary_logloss: 0.00363167
    Early stopping, best iteration is:
    [279]	valid's binary_logloss: 0.00362419
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00448901
    [200]	valid's binary_logloss: 0.00403577
    [300]	valid's binary_logloss: 0.0038792
    [400]	valid's binary_logloss: 0.00381447
    Early stopping, best iteration is:
    [416]	valid's binary_logloss: 0.00380155
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00442546
    [200]	valid's binary_logloss: 0.00400023
    [300]	valid's binary_logloss: 0.00383535
    [400]	valid's binary_logloss: 0.00376333
    Early stopping, best iteration is:
    [469]	valid's binary_logloss: 0.00373504
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00446406
    [200]	valid's binary_logloss: 0.00403455
    [300]	valid's binary_logloss: 0.00386128
    [400]	valid's binary_logloss: 0.0037485
    Early stopping, best iteration is:
    [455]	valid's binary_logloss: 0.00372647
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [45]	valid's binary_logloss: 0.0109033
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [47]	valid's binary_logloss: 0.0108436
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [51]	valid's binary_logloss: 0.0107559
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [51]	valid's binary_logloss: 0.0108025
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [47]	valid's binary_logloss: 0.0108581
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [50]	valid's binary_logloss: 0.0106327
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00572431
    [200]	valid's binary_logloss: 0.00537518
    [300]	valid's binary_logloss: 0.00537516
    [400]	valid's binary_logloss: 0.00537516
    [500]	valid's binary_logloss: 0.00537516
    Early stopping, best iteration is:
    [513]	valid's binary_logloss: 0.00537516
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00570469
    [200]	valid's binary_logloss: 0.00538364
    [300]	valid's binary_logloss: 0.0053836
    [400]	valid's binary_logloss: 0.0053836
    [500]	valid's binary_logloss: 0.0053836
    Early stopping, best iteration is:
    [553]	valid's binary_logloss: 0.0053836
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.0055997
    [200]	valid's binary_logloss: 0.00529033
    [300]	valid's binary_logloss: 0.00529014
    [400]	valid's binary_logloss: 0.00529013
    [500]	valid's binary_logloss: 0.00529013
    Early stopping, best iteration is:
    [563]	valid's binary_logloss: 0.00529013
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00447869
    [200]	valid's binary_logloss: 0.00398079
    [300]	valid's binary_logloss: 0.00384977
    [400]	valid's binary_logloss: 0.00384782
    [500]	valid's binary_logloss: 0.00384782
    [600]	valid's binary_logloss: 0.00384782
    Early stopping, best iteration is:
    [667]	valid's binary_logloss: 0.00384782
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00443943
    [200]	valid's binary_logloss: 0.00394756
    [300]	valid's binary_logloss: 0.00379518
    Early stopping, best iteration is:
    [317]	valid's binary_logloss: 0.0037902
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00437946
    [200]	valid's binary_logloss: 0.00388614
    [300]	valid's binary_logloss: 0.00372758
    Early stopping, best iteration is:
    [320]	valid's binary_logloss: 0.0037234
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [1]	valid's binary_logloss: 0.2404
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [3]	valid's binary_logloss: 0.240396
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [1]	valid's binary_logloss: 0.2404
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [48]	valid's binary_logloss: 0.0107737
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [46]	valid's binary_logloss: 0.0109362
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [50]	valid's binary_logloss: 0.0105068
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00441341
    [200]	valid's binary_logloss: 0.003827
    [300]	valid's binary_logloss: 0.00365045
    [400]	valid's binary_logloss: 0.00360467
    Early stopping, best iteration is:
    [460]	valid's binary_logloss: 0.00359278
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00447238
    [200]	valid's binary_logloss: 0.00387259
    [300]	valid's binary_logloss: 0.00369429
    [400]	valid's binary_logloss: 0.00362241
    [500]	valid's binary_logloss: 0.00359022
    [600]	valid's binary_logloss: 0.00358251
    Early stopping, best iteration is:
    [645]	valid's binary_logloss: 0.00357537
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00439008
    [200]	valid's binary_logloss: 0.00382835
    [300]	valid's binary_logloss: 0.00363157
    [400]	valid's binary_logloss: 0.00354283
    [500]	valid's binary_logloss: 0.00350694
    [600]	valid's binary_logloss: 0.00348788
    [700]	valid's binary_logloss: 0.00348259
    Early stopping, best iteration is:
    [743]	valid's binary_logloss: 0.00347967
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00552169
    [200]	valid's binary_logloss: 0.0053602
    [300]	valid's binary_logloss: 0.0053602
    [400]	valid's binary_logloss: 0.0053602
    Early stopping, best iteration is:
    [379]	valid's binary_logloss: 0.0053602
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00556146
    Early stopping, best iteration is:
    [153]	valid's binary_logloss: 0.00536378
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00560878
    Early stopping, best iteration is:
    [157]	valid's binary_logloss: 0.00536941
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.0049733
    [200]	valid's binary_logloss: 0.00417734
    [300]	valid's binary_logloss: 0.00388546
    [400]	valid's binary_logloss: 0.00373803
    [500]	valid's binary_logloss: 0.00364282
    [600]	valid's binary_logloss: 0.003602
    [700]	valid's binary_logloss: 0.00357665
    [800]	valid's binary_logloss: 0.0035601
    [900]	valid's binary_logloss: 0.0035561
    Early stopping, best iteration is:
    [905]	valid's binary_logloss: 0.00355383
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00500476
    [200]	valid's binary_logloss: 0.00420856
    [300]	valid's binary_logloss: 0.00392999
    [400]	valid's binary_logloss: 0.0037693
    [500]	valid's binary_logloss: 0.00368313
    [600]	valid's binary_logloss: 0.00361975
    [700]	valid's binary_logloss: 0.00358803
    [800]	valid's binary_logloss: 0.00355955
    Early stopping, best iteration is:
    [867]	valid's binary_logloss: 0.00354943
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00494464
    [200]	valid's binary_logloss: 0.00414586
    [300]	valid's binary_logloss: 0.00385391
    [400]	valid's binary_logloss: 0.0037044
    [500]	valid's binary_logloss: 0.00363568
    [600]	valid's binary_logloss: 0.00357033
    [700]	valid's binary_logloss: 0.00352141
    [800]	valid's binary_logloss: 0.00349286
    [900]	valid's binary_logloss: 0.00347933
    [1000]	valid's binary_logloss: 0.00347331
    Early stopping, best iteration is:
    [1048]	valid's binary_logloss: 0.00347092
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [54]	valid's binary_logloss: 0.0106842
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [44]	valid's binary_logloss: 0.0108981
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [48]	valid's binary_logloss: 0.0105272
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [1]	valid's binary_logloss: 0.2404
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [1]	valid's binary_logloss: 0.240396
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [1]	valid's binary_logloss: 0.2404
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00554045
    [200]	valid's binary_logloss: 0.00532964
    [300]	valid's binary_logloss: 0.00532964
    Early stopping, best iteration is:
    [319]	valid's binary_logloss: 0.00532964
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00557092
    [200]	valid's binary_logloss: 0.00538666
    [300]	valid's binary_logloss: 0.00538666
    Early stopping, best iteration is:
    [335]	valid's binary_logloss: 0.00538666
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00553168
    [200]	valid's binary_logloss: 0.00532407
    [300]	valid's binary_logloss: 0.00532407
    Early stopping, best iteration is:
    [311]	valid's binary_logloss: 0.00532407
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00459099
    [200]	valid's binary_logloss: 0.00404461
    [300]	valid's binary_logloss: 0.00383983
    [400]	valid's binary_logloss: 0.00372378
    [500]	valid's binary_logloss: 0.00365511
    [600]	valid's binary_logloss: 0.00360272
    [700]	valid's binary_logloss: 0.00358038
    [800]	valid's binary_logloss: 0.0035542
    [900]	valid's binary_logloss: 0.00353833
    [1000]	valid's binary_logloss: 0.00352034
    Early stopping, best iteration is:
    [1068]	valid's binary_logloss: 0.0035111
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00471412
    [200]	valid's binary_logloss: 0.00415331
    [300]	valid's binary_logloss: 0.00395685
    [400]	valid's binary_logloss: 0.00384643
    [500]	valid's binary_logloss: 0.00377145
    [600]	valid's binary_logloss: 0.0036928
    [700]	valid's binary_logloss: 0.00364695
    [800]	valid's binary_logloss: 0.00361322
    [900]	valid's binary_logloss: 0.00359557
    [1000]	valid's binary_logloss: 0.00357458
    [1100]	valid's binary_logloss: 0.00355634
    Early stopping, best iteration is:
    [1095]	valid's binary_logloss: 0.00355353
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00467624
    [200]	valid's binary_logloss: 0.00411038
    [300]	valid's binary_logloss: 0.00390174
    [400]	valid's binary_logloss: 0.00377782
    [500]	valid's binary_logloss: 0.00369832
    [600]	valid's binary_logloss: 0.00364912
    [700]	valid's binary_logloss: 0.00360699
    [800]	valid's binary_logloss: 0.00357312
    [900]	valid's binary_logloss: 0.00355036
    Early stopping, best iteration is:
    [872]	valid's binary_logloss: 0.00354338
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00411032
    [200]	valid's binary_logloss: 0.0037063
    [300]	valid's binary_logloss: 0.00358867
    Early stopping, best iteration is:
    [360]	valid's binary_logloss: 0.00357237
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00419775
    [200]	valid's binary_logloss: 0.00377779
    [300]	valid's binary_logloss: 0.00366734
    Early stopping, best iteration is:
    [330]	valid's binary_logloss: 0.00364779
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00411815
    [200]	valid's binary_logloss: 0.00368054
    [300]	valid's binary_logloss: 0.00354189
    [400]	valid's binary_logloss: 0.00351434
    Early stopping, best iteration is:
    [427]	valid's binary_logloss: 0.00351023
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.0052085
    [200]	valid's binary_logloss: 0.00442187
    [300]	valid's binary_logloss: 0.00413465
    [400]	valid's binary_logloss: 0.00399793
    [500]	valid's binary_logloss: 0.00392249
    [600]	valid's binary_logloss: 0.00387687
    Early stopping, best iteration is:
    [624]	valid's binary_logloss: 0.003872
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00525917
    [200]	valid's binary_logloss: 0.00447969
    [300]	valid's binary_logloss: 0.00420406
    [400]	valid's binary_logloss: 0.00406794
    [500]	valid's binary_logloss: 0.00398343
    [600]	valid's binary_logloss: 0.00393374
    Early stopping, best iteration is:
    [644]	valid's binary_logloss: 0.00392626
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00520399
    [200]	valid's binary_logloss: 0.00443516
    [300]	valid's binary_logloss: 0.00414034
    [400]	valid's binary_logloss: 0.00400088
    [500]	valid's binary_logloss: 0.00391414
    [600]	valid's binary_logloss: 0.00385436
    Early stopping, best iteration is:
    [664]	valid's binary_logloss: 0.00383779
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.0043908
    [200]	valid's binary_logloss: 0.00400209
    Early stopping, best iteration is:
    [199]	valid's binary_logloss: 0.00400209
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00438484
    [200]	valid's binary_logloss: 0.00398823
    Early stopping, best iteration is:
    [211]	valid's binary_logloss: 0.00398502
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00432784
    [200]	valid's binary_logloss: 0.00394003
    Early stopping, best iteration is:
    [197]	valid's binary_logloss: 0.00393998
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00519781
    [200]	valid's binary_logloss: 0.00439403
    [300]	valid's binary_logloss: 0.00409787
    [400]	valid's binary_logloss: 0.00394198
    [500]	valid's binary_logloss: 0.00383564
    [600]	valid's binary_logloss: 0.00377269
    [700]	valid's binary_logloss: 0.00372602
    [800]	valid's binary_logloss: 0.0036871
    [900]	valid's binary_logloss: 0.00365919
    [1000]	valid's binary_logloss: 0.00363886
    [1100]	valid's binary_logloss: 0.00363015
    [1200]	valid's binary_logloss: 0.00361751
    Early stopping, best iteration is:
    [1214]	valid's binary_logloss: 0.00361532
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00523313
    [200]	valid's binary_logloss: 0.00446466
    [300]	valid's binary_logloss: 0.00417834
    [400]	valid's binary_logloss: 0.00401918
    [500]	valid's binary_logloss: 0.00391413
    [600]	valid's binary_logloss: 0.00385181
    [700]	valid's binary_logloss: 0.00379299
    [800]	valid's binary_logloss: 0.00374342
    [900]	valid's binary_logloss: 0.00371289
    [1000]	valid's binary_logloss: 0.00368482
    Early stopping, best iteration is:
    [1047]	valid's binary_logloss: 0.00367205
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00517763
    [200]	valid's binary_logloss: 0.0043936
    [300]	valid's binary_logloss: 0.00409367
    [400]	valid's binary_logloss: 0.00393958
    [500]	valid's binary_logloss: 0.00383914
    [600]	valid's binary_logloss: 0.00376475
    [700]	valid's binary_logloss: 0.00371103
    [800]	valid's binary_logloss: 0.00366314
    [900]	valid's binary_logloss: 0.00362824
    [1000]	valid's binary_logloss: 0.00360068
    [1100]	valid's binary_logloss: 0.00357615
    [1200]	valid's binary_logloss: 0.00356041
    [1300]	valid's binary_logloss: 0.00355343
    Early stopping, best iteration is:
    [1284]	valid's binary_logloss: 0.00355307
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00400739
    [200]	valid's binary_logloss: 0.00370899
    [300]	valid's binary_logloss: 0.00364508
    Early stopping, best iteration is:
    [356]	valid's binary_logloss: 0.00363543
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00395586
    [200]	valid's binary_logloss: 0.00365366
    [300]	valid's binary_logloss: 0.00357528
    [400]	valid's binary_logloss: 0.00354316
    Early stopping, best iteration is:
    [410]	valid's binary_logloss: 0.00353942
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00397657
    [200]	valid's binary_logloss: 0.00362729
    [300]	valid's binary_logloss: 0.00355328
    Early stopping, best iteration is:
    [332]	valid's binary_logloss: 0.00354348
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00478986
    [200]	valid's binary_logloss: 0.00418216
    [300]	valid's binary_logloss: 0.00396132
    [400]	valid's binary_logloss: 0.00384013
    [500]	valid's binary_logloss: 0.00375629
    [600]	valid's binary_logloss: 0.00371352
    [700]	valid's binary_logloss: 0.00368062
    [800]	valid's binary_logloss: 0.00365512
    Early stopping, best iteration is:
    [864]	valid's binary_logloss: 0.00364202
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00477984
    [200]	valid's binary_logloss: 0.0041826
    [300]	valid's binary_logloss: 0.00395011
    [400]	valid's binary_logloss: 0.00382548
    [500]	valid's binary_logloss: 0.00374847
    [600]	valid's binary_logloss: 0.00368969
    [700]	valid's binary_logloss: 0.00365691
    [800]	valid's binary_logloss: 0.0036206
    [900]	valid's binary_logloss: 0.0035933
    Early stopping, best iteration is:
    [931]	valid's binary_logloss: 0.00358877
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00476331
    [200]	valid's binary_logloss: 0.00413691
    [300]	valid's binary_logloss: 0.00390148
    [400]	valid's binary_logloss: 0.00378442
    [500]	valid's binary_logloss: 0.00369253
    [600]	valid's binary_logloss: 0.00364266
    [700]	valid's binary_logloss: 0.00359904
    [800]	valid's binary_logloss: 0.00357109
    [900]	valid's binary_logloss: 0.0035511
    Early stopping, best iteration is:
    [914]	valid's binary_logloss: 0.00354393
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00492221
    [200]	valid's binary_logloss: 0.00437419
    [300]	valid's binary_logloss: 0.00415729
    [400]	valid's binary_logloss: 0.00403889
    [500]	valid's binary_logloss: 0.00396618
    Early stopping, best iteration is:
    [536]	valid's binary_logloss: 0.00395034
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00499761
    [200]	valid's binary_logloss: 0.00442101
    [300]	valid's binary_logloss: 0.004231
    [400]	valid's binary_logloss: 0.00413273
    [500]	valid's binary_logloss: 0.00406268
    [600]	valid's binary_logloss: 0.00401665
    Early stopping, best iteration is:
    [571]	valid's binary_logloss: 0.00401665
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00497218
    [200]	valid's binary_logloss: 0.00435987
    [300]	valid's binary_logloss: 0.00417188
    [400]	valid's binary_logloss: 0.0040496
    [500]	valid's binary_logloss: 0.00398055
    Early stopping, best iteration is:
    [562]	valid's binary_logloss: 0.00395228
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00457243
    [200]	valid's binary_logloss: 0.00390266
    [300]	valid's binary_logloss: 0.00369636
    [400]	valid's binary_logloss: 0.00359621
    [500]	valid's binary_logloss: 0.00356151
    [600]	valid's binary_logloss: 0.00354669
    Early stopping, best iteration is:
    [621]	valid's binary_logloss: 0.00354091
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00461384
    [200]	valid's binary_logloss: 0.00397932
    [300]	valid's binary_logloss: 0.00377863
    [400]	valid's binary_logloss: 0.00368547
    [500]	valid's binary_logloss: 0.00364227
    [600]	valid's binary_logloss: 0.00361437
    [700]	valid's binary_logloss: 0.00359836
    Early stopping, best iteration is:
    [754]	valid's binary_logloss: 0.0035911
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00452504
    [200]	valid's binary_logloss: 0.00388757
    [300]	valid's binary_logloss: 0.00368631
    [400]	valid's binary_logloss: 0.00358491
    [500]	valid's binary_logloss: 0.00352217
    [600]	valid's binary_logloss: 0.00348442
    [700]	valid's binary_logloss: 0.00346527
    [800]	valid's binary_logloss: 0.00344887
    Early stopping, best iteration is:
    [804]	valid's binary_logloss: 0.00344767
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00556947
    Early stopping, best iteration is:
    [147]	valid's binary_logloss: 0.00541327
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00555832
    Early stopping, best iteration is:
    [153]	valid's binary_logloss: 0.0053794
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00546684
    Early stopping, best iteration is:
    [151]	valid's binary_logloss: 0.00525947
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00497395
    [200]	valid's binary_logloss: 0.00441719
    [300]	valid's binary_logloss: 0.00422596
    [400]	valid's binary_logloss: 0.00413601
    Early stopping, best iteration is:
    [377]	valid's binary_logloss: 0.00413601
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00497943
    [200]	valid's binary_logloss: 0.00442532
    [300]	valid's binary_logloss: 0.00425463
    Early stopping, best iteration is:
    [345]	valid's binary_logloss: 0.00421225
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00494881
    [200]	valid's binary_logloss: 0.0044145
    [300]	valid's binary_logloss: 0.00422156
    Early stopping, best iteration is:
    [356]	valid's binary_logloss: 0.00415749
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00506413
    [200]	valid's binary_logloss: 0.00424572
    [300]	valid's binary_logloss: 0.00395271
    [400]	valid's binary_logloss: 0.0038057
    [500]	valid's binary_logloss: 0.00371318
    [600]	valid's binary_logloss: 0.00364191
    [700]	valid's binary_logloss: 0.00360131
    [800]	valid's binary_logloss: 0.00356803
    [900]	valid's binary_logloss: 0.00354639
    Early stopping, best iteration is:
    [948]	valid's binary_logloss: 0.00353655
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00510775
    [200]	valid's binary_logloss: 0.00431801
    [300]	valid's binary_logloss: 0.00403104
    [400]	valid's binary_logloss: 0.00386644
    [500]	valid's binary_logloss: 0.00377417
    [600]	valid's binary_logloss: 0.00370019
    [700]	valid's binary_logloss: 0.00364697
    [800]	valid's binary_logloss: 0.00360722
    [900]	valid's binary_logloss: 0.0035796
    [1000]	valid's binary_logloss: 0.0035595
    [1100]	valid's binary_logloss: 0.00354119
    [1200]	valid's binary_logloss: 0.00352938
    [1300]	valid's binary_logloss: 0.00352071
    Early stopping, best iteration is:
    [1348]	valid's binary_logloss: 0.00351487
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00503648
    [200]	valid's binary_logloss: 0.00422855
    [300]	valid's binary_logloss: 0.0039443
    [400]	valid's binary_logloss: 0.00379084
    [500]	valid's binary_logloss: 0.00369029
    [600]	valid's binary_logloss: 0.00361541
    [700]	valid's binary_logloss: 0.00356154
    [800]	valid's binary_logloss: 0.00352094
    [900]	valid's binary_logloss: 0.00349343
    [1000]	valid's binary_logloss: 0.00346435
    [1100]	valid's binary_logloss: 0.00344755
    [1200]	valid's binary_logloss: 0.00343436
    [1300]	valid's binary_logloss: 0.00342217
    [1400]	valid's binary_logloss: 0.00341643
    Early stopping, best iteration is:
    [1454]	valid's binary_logloss: 0.00341296
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00529652
    [200]	valid's binary_logloss: 0.00462806
    [300]	valid's binary_logloss: 0.00437119
    [400]	valid's binary_logloss: 0.00423889
    [500]	valid's binary_logloss: 0.00415096
    [600]	valid's binary_logloss: 0.00408077
    [700]	valid's binary_logloss: 0.00401294
    [800]	valid's binary_logloss: 0.00396437
    [900]	valid's binary_logloss: 0.00393248
    [1000]	valid's binary_logloss: 0.00389526
    [1100]	valid's binary_logloss: 0.0038716
    [1200]	valid's binary_logloss: 0.00385202
    [1300]	valid's binary_logloss: 0.00384128
    Early stopping, best iteration is:
    [1271]	valid's binary_logloss: 0.00384091
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00536723
    [200]	valid's binary_logloss: 0.00467427
    [300]	valid's binary_logloss: 0.00443057
    [400]	valid's binary_logloss: 0.00431336
    [500]	valid's binary_logloss: 0.00422967
    [600]	valid's binary_logloss: 0.00416189
    [700]	valid's binary_logloss: 0.0041149
    [800]	valid's binary_logloss: 0.00406871
    [900]	valid's binary_logloss: 0.00403917
    [1000]	valid's binary_logloss: 0.00400062
    [1100]	valid's binary_logloss: 0.00397244
    [1200]	valid's binary_logloss: 0.00394052
    [1300]	valid's binary_logloss: 0.00392912
    Early stopping, best iteration is:
    [1347]	valid's binary_logloss: 0.00392508
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00529169
    [200]	valid's binary_logloss: 0.00458567
    [300]	valid's binary_logloss: 0.0043293
    [400]	valid's binary_logloss: 0.00419885
    [500]	valid's binary_logloss: 0.00410902
    [600]	valid's binary_logloss: 0.00403967
    [700]	valid's binary_logloss: 0.00398495
    [800]	valid's binary_logloss: 0.00393278
    [900]	valid's binary_logloss: 0.00389649
    [1000]	valid's binary_logloss: 0.00385665
    [1100]	valid's binary_logloss: 0.00383147
    [1200]	valid's binary_logloss: 0.00380137
    [1300]	valid's binary_logloss: 0.00377898
    Early stopping, best iteration is:
    [1335]	valid's binary_logloss: 0.00377107
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00504084
    [200]	valid's binary_logloss: 0.00437724
    [300]	valid's binary_logloss: 0.00413131
    [400]	valid's binary_logloss: 0.00398152
    [500]	valid's binary_logloss: 0.00389449
    [600]	valid's binary_logloss: 0.00382199
    [700]	valid's binary_logloss: 0.00377292
    [800]	valid's binary_logloss: 0.00373002
    [900]	valid's binary_logloss: 0.00370718
    Early stopping, best iteration is:
    [963]	valid's binary_logloss: 0.00368693
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.0050892
    [200]	valid's binary_logloss: 0.00439799
    [300]	valid's binary_logloss: 0.00416138
    [400]	valid's binary_logloss: 0.00403099
    [500]	valid's binary_logloss: 0.00395143
    [600]	valid's binary_logloss: 0.00388425
    [700]	valid's binary_logloss: 0.00381784
    [800]	valid's binary_logloss: 0.00376494
    [900]	valid's binary_logloss: 0.00372539
    [1000]	valid's binary_logloss: 0.00370013
    [1100]	valid's binary_logloss: 0.0036695
    [1200]	valid's binary_logloss: 0.00364888
    Early stopping, best iteration is:
    [1214]	valid's binary_logloss: 0.00364297
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00512689
    [200]	valid's binary_logloss: 0.00440895
    [300]	valid's binary_logloss: 0.00418371
    [400]	valid's binary_logloss: 0.00403119
    [500]	valid's binary_logloss: 0.00393368
    [600]	valid's binary_logloss: 0.00386388
    [700]	valid's binary_logloss: 0.00380691
    [800]	valid's binary_logloss: 0.00377325
    [900]	valid's binary_logloss: 0.00372367
    Early stopping, best iteration is:
    [922]	valid's binary_logloss: 0.00370968
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.0057053
    [200]	valid's binary_logloss: 0.00526054
    [300]	valid's binary_logloss: 0.00525917
    [400]	valid's binary_logloss: 0.00525916
    [500]	valid's binary_logloss: 0.00525916
    [600]	valid's binary_logloss: 0.00525916
    [700]	valid's binary_logloss: 0.00525916
    [800]	valid's binary_logloss: 0.00525916
    [900]	valid's binary_logloss: 0.00525916
    Early stopping, best iteration is:
    [968]	valid's binary_logloss: 0.00525916
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00579103
    [200]	valid's binary_logloss: 0.00535062
    [300]	valid's binary_logloss: 0.00534163
    [400]	valid's binary_logloss: 0.00534162
    [500]	valid's binary_logloss: 0.00534162
    [600]	valid's binary_logloss: 0.00534162
    [700]	valid's binary_logloss: 0.00534162
    [800]	valid's binary_logloss: 0.00534162
    Early stopping, best iteration is:
    [864]	valid's binary_logloss: 0.00534162
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00565571
    [200]	valid's binary_logloss: 0.00522563
    [300]	valid's binary_logloss: 0.00522265
    [400]	valid's binary_logloss: 0.00522264
    [500]	valid's binary_logloss: 0.00522264
    [600]	valid's binary_logloss: 0.00522264
    Early stopping, best iteration is:
    [599]	valid's binary_logloss: 0.00522264
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [2]	valid's binary_logloss: 0.2404
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [1]	valid's binary_logloss: 0.240396
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [3]	valid's binary_logloss: 0.2404
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [49]	valid's binary_logloss: 0.0110207
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [47]	valid's binary_logloss: 0.0109958
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [53]	valid's binary_logloss: 0.0105991
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00634662
    [200]	valid's binary_logloss: 0.00582439
    Early stopping, best iteration is:
    [260]	valid's binary_logloss: 0.00575149
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00631152
    [200]	valid's binary_logloss: 0.00583064
    [300]	valid's binary_logloss: 0.0057147
    Early stopping, best iteration is:
    [292]	valid's binary_logloss: 0.0057147
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00624726
    [200]	valid's binary_logloss: 0.00578799
    [300]	valid's binary_logloss: 0.00569062
    [400]	valid's binary_logloss: 0.00569062
    [500]	valid's binary_logloss: 0.00569062
    [600]	valid's binary_logloss: 0.00569062
    [700]	valid's binary_logloss: 0.00569062
    Early stopping, best iteration is:
    [672]	valid's binary_logloss: 0.00569062
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00456946
    [200]	valid's binary_logloss: 0.0039584
    [300]	valid's binary_logloss: 0.00372626
    [400]	valid's binary_logloss: 0.00362128
    [500]	valid's binary_logloss: 0.00355972
    Early stopping, best iteration is:
    [498]	valid's binary_logloss: 0.00355916
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00454799
    [200]	valid's binary_logloss: 0.00397455
    [300]	valid's binary_logloss: 0.00377669
    [400]	valid's binary_logloss: 0.00367076
    [500]	valid's binary_logloss: 0.00360142
    [600]	valid's binary_logloss: 0.00355434
    [700]	valid's binary_logloss: 0.00352271
    [800]	valid's binary_logloss: 0.00350334
    Early stopping, best iteration is:
    [826]	valid's binary_logloss: 0.00349051
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00453882
    [200]	valid's binary_logloss: 0.00394762
    [300]	valid's binary_logloss: 0.00371796
    [400]	valid's binary_logloss: 0.00361363
    [500]	valid's binary_logloss: 0.00354338
    [600]	valid's binary_logloss: 0.00350701
    [700]	valid's binary_logloss: 0.00347692
    [800]	valid's binary_logloss: 0.00346415
    Early stopping, best iteration is:
    [869]	valid's binary_logloss: 0.00345551
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [47]	valid's binary_logloss: 0.011055
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [47]	valid's binary_logloss: 0.0110067
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [51]	valid's binary_logloss: 0.010674
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00642685
    [200]	valid's binary_logloss: 0.0058676
    [300]	valid's binary_logloss: 0.00571022
    [400]	valid's binary_logloss: 0.00569492
    [500]	valid's binary_logloss: 0.0056949
    [600]	valid's binary_logloss: 0.0056949
    [700]	valid's binary_logloss: 0.0056949
    [800]	valid's binary_logloss: 0.0056949
    Early stopping, best iteration is:
    [791]	valid's binary_logloss: 0.0056949
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00641047
    [200]	valid's binary_logloss: 0.00587056
    [300]	valid's binary_logloss: 0.0057088
    [400]	valid's binary_logloss: 0.00567562
    [500]	valid's binary_logloss: 0.00567562
    [600]	valid's binary_logloss: 0.00567562
    [700]	valid's binary_logloss: 0.00567562
    [800]	valid's binary_logloss: 0.00567562
    [900]	valid's binary_logloss: 0.00567562
    [1000]	valid's binary_logloss: 0.00567562
    [1100]	valid's binary_logloss: 0.00567562
    [1200]	valid's binary_logloss: 0.00567562
    Early stopping, best iteration is:
    [1268]	valid's binary_logloss: 0.00567562
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00635987
    [200]	valid's binary_logloss: 0.0058264
    [300]	valid's binary_logloss: 0.00565312
    [400]	valid's binary_logloss: 0.00562212
    [500]	valid's binary_logloss: 0.00562208
    [600]	valid's binary_logloss: 0.00562208
    [700]	valid's binary_logloss: 0.00562208
    [800]	valid's binary_logloss: 0.00562208
    [900]	valid's binary_logloss: 0.00562208
    Early stopping, best iteration is:
    [957]	valid's binary_logloss: 0.00562208
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00515928
    [200]	valid's binary_logloss: 0.00436751
    [300]	valid's binary_logloss: 0.00410067
    [400]	valid's binary_logloss: 0.00396993
    [500]	valid's binary_logloss: 0.00388773
    Early stopping, best iteration is:
    [556]	valid's binary_logloss: 0.00386631
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00517768
    [200]	valid's binary_logloss: 0.00442144
    [300]	valid's binary_logloss: 0.00416728
    [400]	valid's binary_logloss: 0.00404468
    [500]	valid's binary_logloss: 0.00397452
    Early stopping, best iteration is:
    [544]	valid's binary_logloss: 0.00395584
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00512595
    [200]	valid's binary_logloss: 0.00436108
    [300]	valid's binary_logloss: 0.00408081
    [400]	valid's binary_logloss: 0.0039534
    [500]	valid's binary_logloss: 0.00387777
    [600]	valid's binary_logloss: 0.00383748
    Early stopping, best iteration is:
    [588]	valid's binary_logloss: 0.00383748
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00471206
    [200]	valid's binary_logloss: 0.00403082
    [300]	valid's binary_logloss: 0.0037948
    [400]	valid's binary_logloss: 0.00367523
    [500]	valid's binary_logloss: 0.00362342
    [600]	valid's binary_logloss: 0.00358538
    [700]	valid's binary_logloss: 0.00357039
    [800]	valid's binary_logloss: 0.00356127
    Early stopping, best iteration is:
    [783]	valid's binary_logloss: 0.00355924
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00477195
    [200]	valid's binary_logloss: 0.00408765
    [300]	valid's binary_logloss: 0.00386927
    [400]	valid's binary_logloss: 0.00376337
    [500]	valid's binary_logloss: 0.00369177
    [600]	valid's binary_logloss: 0.00364344
    [700]	valid's binary_logloss: 0.00362057
    [800]	valid's binary_logloss: 0.00359836
    [900]	valid's binary_logloss: 0.00357983
    Early stopping, best iteration is:
    [951]	valid's binary_logloss: 0.00357338
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.004679
    [200]	valid's binary_logloss: 0.00403006
    [300]	valid's binary_logloss: 0.00378681
    [400]	valid's binary_logloss: 0.00367375
    [500]	valid's binary_logloss: 0.00360339
    [600]	valid's binary_logloss: 0.00355272
    [700]	valid's binary_logloss: 0.00350515
    [800]	valid's binary_logloss: 0.00348627
    Early stopping, best iteration is:
    [818]	valid's binary_logloss: 0.00348126
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00562036
    [200]	valid's binary_logloss: 0.00532989
    [300]	valid's binary_logloss: 0.00532989
    [400]	valid's binary_logloss: 0.00532989
    Early stopping, best iteration is:
    [401]	valid's binary_logloss: 0.00532989
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00563523
    [200]	valid's binary_logloss: 0.00536728
    [300]	valid's binary_logloss: 0.00536727
    [400]	valid's binary_logloss: 0.00536727
    [500]	valid's binary_logloss: 0.00536727
    [600]	valid's binary_logloss: 0.00536727
    Early stopping, best iteration is:
    [577]	valid's binary_logloss: 0.00536727
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00557317
    Early stopping, best iteration is:
    [163]	valid's binary_logloss: 0.00529023
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [45]	valid's binary_logloss: 0.0109362
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [45]	valid's binary_logloss: 0.0108481
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [48]	valid's binary_logloss: 0.0108449
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00452054
    [200]	valid's binary_logloss: 0.00405886
    [300]	valid's binary_logloss: 0.00387893
    [400]	valid's binary_logloss: 0.00382281
    Early stopping, best iteration is:
    [391]	valid's binary_logloss: 0.00382213
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00452482
    [200]	valid's binary_logloss: 0.00407155
    [300]	valid's binary_logloss: 0.00394468
    Early stopping, best iteration is:
    [361]	valid's binary_logloss: 0.00390293
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00452756
    [200]	valid's binary_logloss: 0.00402852
    [300]	valid's binary_logloss: 0.00388541
    Early stopping, best iteration is:
    [366]	valid's binary_logloss: 0.00383804
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00544572
    [200]	valid's binary_logloss: 0.00505059
    [300]	valid's binary_logloss: 0.00490183
    [400]	valid's binary_logloss: 0.00480251
    [500]	valid's binary_logloss: 0.00474566
    [600]	valid's binary_logloss: 0.00470033
    [700]	valid's binary_logloss: 0.00467073
    [800]	valid's binary_logloss: 0.00463247
    Early stopping, best iteration is:
    [830]	valid's binary_logloss: 0.00462532
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00543634
    [200]	valid's binary_logloss: 0.00505544
    [300]	valid's binary_logloss: 0.00492519
    [400]	valid's binary_logloss: 0.00482707
    [500]	valid's binary_logloss: 0.00476826
    [600]	valid's binary_logloss: 0.00472581
    [700]	valid's binary_logloss: 0.00471028
    [800]	valid's binary_logloss: 0.00469342
    Early stopping, best iteration is:
    [784]	valid's binary_logloss: 0.00469326
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00534558
    [200]	valid's binary_logloss: 0.00497033
    [300]	valid's binary_logloss: 0.00481825
    [400]	valid's binary_logloss: 0.00471693
    [500]	valid's binary_logloss: 0.00464792
    [600]	valid's binary_logloss: 0.00461325
    [700]	valid's binary_logloss: 0.00457825
    Early stopping, best iteration is:
    [763]	valid's binary_logloss: 0.00456102
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00630322
    [200]	valid's binary_logloss: 0.00579156
    Early stopping, best iteration is:
    [265]	valid's binary_logloss: 0.0057154
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00623966
    [200]	valid's binary_logloss: 0.0058016
    [300]	valid's binary_logloss: 0.00570686
    [400]	valid's binary_logloss: 0.00570686
    Early stopping, best iteration is:
    [460]	valid's binary_logloss: 0.00570686
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.0062272
    [200]	valid's binary_logloss: 0.00574761
    [300]	valid's binary_logloss: 0.00566556
    Early stopping, best iteration is:
    [337]	valid's binary_logloss: 0.00566548
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00483522
    [200]	valid's binary_logloss: 0.00421031
    [300]	valid's binary_logloss: 0.00396214
    [400]	valid's binary_logloss: 0.00385218
    [500]	valid's binary_logloss: 0.00378673
    [600]	valid's binary_logloss: 0.00373637
    [700]	valid's binary_logloss: 0.00369644
    [800]	valid's binary_logloss: 0.00367047
    [900]	valid's binary_logloss: 0.00364187
    Early stopping, best iteration is:
    [969]	valid's binary_logloss: 0.00363098
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00476813
    [200]	valid's binary_logloss: 0.00415924
    [300]	valid's binary_logloss: 0.00394526
    [400]	valid's binary_logloss: 0.00383608
    [500]	valid's binary_logloss: 0.00375552
    [600]	valid's binary_logloss: 0.00369475
    [700]	valid's binary_logloss: 0.00365996
    [800]	valid's binary_logloss: 0.00363319
    [900]	valid's binary_logloss: 0.00360398
    [1000]	valid's binary_logloss: 0.00358579
    Early stopping, best iteration is:
    [1011]	valid's binary_logloss: 0.00358423
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00475796
    [200]	valid's binary_logloss: 0.00414212
    [300]	valid's binary_logloss: 0.00391085
    [400]	valid's binary_logloss: 0.00379205
    [500]	valid's binary_logloss: 0.00371566
    [600]	valid's binary_logloss: 0.00365957
    [700]	valid's binary_logloss: 0.00361345
    [800]	valid's binary_logloss: 0.0035816
    [900]	valid's binary_logloss: 0.00356558
    [1000]	valid's binary_logloss: 0.00355489
    Early stopping, best iteration is:
    [1044]	valid's binary_logloss: 0.0035505
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00429395
    [200]	valid's binary_logloss: 0.00395854
    Early stopping, best iteration is:
    [186]	valid's binary_logloss: 0.0039584
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00429327
    [200]	valid's binary_logloss: 0.00392876
    Early stopping, best iteration is:
    [212]	valid's binary_logloss: 0.00391977
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00429131
    [200]	valid's binary_logloss: 0.00396117
    Early stopping, best iteration is:
    [193]	valid's binary_logloss: 0.00396086
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00394633
    [200]	valid's binary_logloss: 0.00362296
    Early stopping, best iteration is:
    [220]	valid's binary_logloss: 0.00360385
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00389502
    [200]	valid's binary_logloss: 0.00351461
    [300]	valid's binary_logloss: 0.00345873
    Early stopping, best iteration is:
    [284]	valid's binary_logloss: 0.00345004
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00389606
    [200]	valid's binary_logloss: 0.00356637
    Early stopping, best iteration is:
    [268]	valid's binary_logloss: 0.00350192
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00535532
    [200]	valid's binary_logloss: 0.00497804
    [300]	valid's binary_logloss: 0.00482409
    [400]	valid's binary_logloss: 0.00472937
    [500]	valid's binary_logloss: 0.00466686
    [600]	valid's binary_logloss: 0.00461839
    [700]	valid's binary_logloss: 0.0045883
    [800]	valid's binary_logloss: 0.00456715
    [900]	valid's binary_logloss: 0.00454852
    [1000]	valid's binary_logloss: 0.0045362
    Early stopping, best iteration is:
    [983]	valid's binary_logloss: 0.00453583
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00542142
    [200]	valid's binary_logloss: 0.0050317
    [300]	valid's binary_logloss: 0.00488705
    [400]	valid's binary_logloss: 0.00479085
    [500]	valid's binary_logloss: 0.00473248
    [600]	valid's binary_logloss: 0.00469272
    [700]	valid's binary_logloss: 0.0046712
    [800]	valid's binary_logloss: 0.00464472
    [900]	valid's binary_logloss: 0.00462657
    [1000]	valid's binary_logloss: 0.00460828
    [1100]	valid's binary_logloss: 0.00459818
    [1200]	valid's binary_logloss: 0.00459068
    [1300]	valid's binary_logloss: 0.00457733
    [1400]	valid's binary_logloss: 0.00457113
    Early stopping, best iteration is:
    [1388]	valid's binary_logloss: 0.00457081
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00531348
    [200]	valid's binary_logloss: 0.00494962
    [300]	valid's binary_logloss: 0.00476835
    [400]	valid's binary_logloss: 0.00465848
    [500]	valid's binary_logloss: 0.00458846
    [600]	valid's binary_logloss: 0.00454202
    [700]	valid's binary_logloss: 0.00449777
    [800]	valid's binary_logloss: 0.00446871
    [900]	valid's binary_logloss: 0.00443757
    [1000]	valid's binary_logloss: 0.00442022
    [1100]	valid's binary_logloss: 0.0044048
    [1200]	valid's binary_logloss: 0.00438989
    Early stopping, best iteration is:
    [1236]	valid's binary_logloss: 0.00438238
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00465988
    [200]	valid's binary_logloss: 0.00414207
    [300]	valid's binary_logloss: 0.00400224
    Early stopping, best iteration is:
    [280]	valid's binary_logloss: 0.00400224
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00462537
    [200]	valid's binary_logloss: 0.00412019
    Early stopping, best iteration is:
    [253]	valid's binary_logloss: 0.00404343
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00458645
    [200]	valid's binary_logloss: 0.00409322
    [300]	valid's binary_logloss: 0.00398729
    Early stopping, best iteration is:
    [275]	valid's binary_logloss: 0.00398729
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00495409
    [200]	valid's binary_logloss: 0.00425031
    [300]	valid's binary_logloss: 0.0040169
    [400]	valid's binary_logloss: 0.00391404
    Early stopping, best iteration is:
    [427]	valid's binary_logloss: 0.00390007
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00499354
    [200]	valid's binary_logloss: 0.00432774
    [300]	valid's binary_logloss: 0.00409899
    [400]	valid's binary_logloss: 0.00399058
    Early stopping, best iteration is:
    [419]	valid's binary_logloss: 0.00397899
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00492989
    [200]	valid's binary_logloss: 0.00426071
    [300]	valid's binary_logloss: 0.00402856
    [400]	valid's binary_logloss: 0.00391421
    Early stopping, best iteration is:
    [433]	valid's binary_logloss: 0.00390679
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00428072
    [200]	valid's binary_logloss: 0.00382439
    [300]	valid's binary_logloss: 0.00368095
    [400]	valid's binary_logloss: 0.00363643
    Early stopping, best iteration is:
    [447]	valid's binary_logloss: 0.00362775
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00423229
    [200]	valid's binary_logloss: 0.00378924
    [300]	valid's binary_logloss: 0.00365878
    [400]	valid's binary_logloss: 0.00360481
    Early stopping, best iteration is:
    [408]	valid's binary_logloss: 0.00360065
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00422878
    [200]	valid's binary_logloss: 0.00378168
    [300]	valid's binary_logloss: 0.0036417
    [400]	valid's binary_logloss: 0.00357682
    [500]	valid's binary_logloss: 0.0035549
    Early stopping, best iteration is:
    [516]	valid's binary_logloss: 0.00354783
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00424109
    [200]	valid's binary_logloss: 0.00376827
    [300]	valid's binary_logloss: 0.00361698
    [400]	valid's binary_logloss: 0.00355619
    Early stopping, best iteration is:
    [444]	valid's binary_logloss: 0.00355086
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00428769
    [200]	valid's binary_logloss: 0.00381914
    [300]	valid's binary_logloss: 0.00365125
    [400]	valid's binary_logloss: 0.00357917
    [500]	valid's binary_logloss: 0.00354244
    Early stopping, best iteration is:
    [516]	valid's binary_logloss: 0.00353581
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00426733
    [200]	valid's binary_logloss: 0.00382445
    [300]	valid's binary_logloss: 0.00364031
    [400]	valid's binary_logloss: 0.00355676
    [500]	valid's binary_logloss: 0.00351773
    Early stopping, best iteration is:
    [482]	valid's binary_logloss: 0.00351396
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00527456
    [200]	valid's binary_logloss: 0.00487495
    [300]	valid's binary_logloss: 0.00473973
    [400]	valid's binary_logloss: 0.00465855
    [500]	valid's binary_logloss: 0.00459215
    [600]	valid's binary_logloss: 0.00455414
    [700]	valid's binary_logloss: 0.00450196
    [800]	valid's binary_logloss: 0.00447329
    [900]	valid's binary_logloss: 0.00446028
    [1000]	valid's binary_logloss: 0.0044241
    [1100]	valid's binary_logloss: 0.0044165
    [1200]	valid's binary_logloss: 0.00439312
    [1300]	valid's binary_logloss: 0.00436805
    [1400]	valid's binary_logloss: 0.00436074
    Early stopping, best iteration is:
    [1446]	valid's binary_logloss: 0.00435728
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00525219
    [200]	valid's binary_logloss: 0.00484468
    [300]	valid's binary_logloss: 0.00470627
    [400]	valid's binary_logloss: 0.00461873
    [500]	valid's binary_logloss: 0.00454885
    [600]	valid's binary_logloss: 0.00449148
    [700]	valid's binary_logloss: 0.00445069
    [800]	valid's binary_logloss: 0.00443028
    Early stopping, best iteration is:
    [794]	valid's binary_logloss: 0.00442881
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00527202
    [200]	valid's binary_logloss: 0.00486281
    [300]	valid's binary_logloss: 0.00467748
    [400]	valid's binary_logloss: 0.00457592
    [500]	valid's binary_logloss: 0.00449337
    [600]	valid's binary_logloss: 0.00444176
    [700]	valid's binary_logloss: 0.00440719
    [800]	valid's binary_logloss: 0.00436028
    [900]	valid's binary_logloss: 0.00433735
    [1000]	valid's binary_logloss: 0.00431966
    [1100]	valid's binary_logloss: 0.00430175
    [1200]	valid's binary_logloss: 0.00428383
    Early stopping, best iteration is:
    [1242]	valid's binary_logloss: 0.00427111
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00525913
    [200]	valid's binary_logloss: 0.00456871
    [300]	valid's binary_logloss: 0.00430933
    [400]	valid's binary_logloss: 0.00418566
    [500]	valid's binary_logloss: 0.00411007
    [600]	valid's binary_logloss: 0.00406068
    Early stopping, best iteration is:
    [589]	valid's binary_logloss: 0.00406029
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.0053168
    [200]	valid's binary_logloss: 0.00461953
    [300]	valid's binary_logloss: 0.00437785
    [400]	valid's binary_logloss: 0.00424763
    [500]	valid's binary_logloss: 0.00416936
    [600]	valid's binary_logloss: 0.00410765
    Early stopping, best iteration is:
    [616]	valid's binary_logloss: 0.00410606
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00523776
    [200]	valid's binary_logloss: 0.0045646
    [300]	valid's binary_logloss: 0.00432525
    [400]	valid's binary_logloss: 0.00418184
    [500]	valid's binary_logloss: 0.00408894
    Early stopping, best iteration is:
    [553]	valid's binary_logloss: 0.00405957
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [2]	valid's binary_logloss: 0.2404
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [1]	valid's binary_logloss: 0.240396
    Training until validation scores don't improve for 30 rounds.
    Early stopping, best iteration is:
    [1]	valid's binary_logloss: 0.2404
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00434698
    [200]	valid's binary_logloss: 0.00391124
    [300]	valid's binary_logloss: 0.00379302
    Early stopping, best iteration is:
    [331]	valid's binary_logloss: 0.00377855
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00432176
    [200]	valid's binary_logloss: 0.00390344
    [300]	valid's binary_logloss: 0.0037804
    [400]	valid's binary_logloss: 0.00374419
    Early stopping, best iteration is:
    [370]	valid's binary_logloss: 0.00374276
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00427591
    [200]	valid's binary_logloss: 0.00385637
    [300]	valid's binary_logloss: 0.0037213
    Early stopping, best iteration is:
    [353]	valid's binary_logloss: 0.00366867
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00447095
    [200]	valid's binary_logloss: 0.00396158
    [300]	valid's binary_logloss: 0.00380019
    [400]	valid's binary_logloss: 0.00371002
    [500]	valid's binary_logloss: 0.00367417
    [600]	valid's binary_logloss: 0.0036466
    Early stopping, best iteration is:
    [622]	valid's binary_logloss: 0.00364147
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00444596
    [200]	valid's binary_logloss: 0.00398984
    [300]	valid's binary_logloss: 0.00381747
    [400]	valid's binary_logloss: 0.00372617
    [500]	valid's binary_logloss: 0.00366548
    [600]	valid's binary_logloss: 0.00364244
    Early stopping, best iteration is:
    [665]	valid's binary_logloss: 0.00362444
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00445929
    [200]	valid's binary_logloss: 0.00397015
    [300]	valid's binary_logloss: 0.00380161
    [400]	valid's binary_logloss: 0.0037008
    [500]	valid's binary_logloss: 0.00362627
    [600]	valid's binary_logloss: 0.00359725
    [700]	valid's binary_logloss: 0.00357264
    Early stopping, best iteration is:
    [713]	valid's binary_logloss: 0.00356762
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00400357
    [200]	valid's binary_logloss: 0.00364645
    [300]	valid's binary_logloss: 0.00357222
    Early stopping, best iteration is:
    [316]	valid's binary_logloss: 0.00355641
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00402175
    [200]	valid's binary_logloss: 0.00359004
    [300]	valid's binary_logloss: 0.00348911
    Early stopping, best iteration is:
    [302]	valid's binary_logloss: 0.0034881
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00397279
    [200]	valid's binary_logloss: 0.00359919
    [300]	valid's binary_logloss: 0.0034953
    Early stopping, best iteration is:
    [302]	valid's binary_logloss: 0.00349015
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.0047837
    [200]	valid's binary_logloss: 0.00420712
    [300]	valid's binary_logloss: 0.00399997
    [400]	valid's binary_logloss: 0.00389178
    [500]	valid's binary_logloss: 0.00382901
    [600]	valid's binary_logloss: 0.00377989
    Early stopping, best iteration is:
    [668]	valid's binary_logloss: 0.00375702
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00475682
    [200]	valid's binary_logloss: 0.00418564
    [300]	valid's binary_logloss: 0.00399324
    [400]	valid's binary_logloss: 0.00388597
    [500]	valid's binary_logloss: 0.00382648
    [600]	valid's binary_logloss: 0.00376451
    [700]	valid's binary_logloss: 0.00373179
    Early stopping, best iteration is:
    [737]	valid's binary_logloss: 0.00371999
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.0047856
    [200]	valid's binary_logloss: 0.00418688
    [300]	valid's binary_logloss: 0.0040104
    [400]	valid's binary_logloss: 0.0038844
    [500]	valid's binary_logloss: 0.00381392
    [600]	valid's binary_logloss: 0.00376405
    [700]	valid's binary_logloss: 0.00372318
    [800]	valid's binary_logloss: 0.00369658
    Early stopping, best iteration is:
    [786]	valid's binary_logloss: 0.00369658


    [Parallel(n_jobs=1)]: Done 300 out of 300 | elapsed: 249.9min finished


    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00468361
    [200]	valid's binary_logloss: 0.00392183
    [300]	valid's binary_logloss: 0.00364829
    [400]	valid's binary_logloss: 0.00348944
    [500]	valid's binary_logloss: 0.00337939
    [600]	valid's binary_logloss: 0.00330502
    [700]	valid's binary_logloss: 0.00325775
    [800]	valid's binary_logloss: 0.00322144
    [900]	valid's binary_logloss: 0.00319789
    [1000]	valid's binary_logloss: 0.0031711
    [1100]	valid's binary_logloss: 0.00315203
    Early stopping, best iteration is:
    [1157]	valid's binary_logloss: 0.00314025
    Best score reached: 0.9988364249236769 with params: {'colsample_bytree': 0.9501241488957805, 'min_child_samples': 301, 'min_child_weight': 0.1, 'num_leaves': 28, 'reg_alpha': 0, 'reg_lambda': 100, 'subsample': 0.9326466073236168} 
    Fitting 3 folds for each of 4 candidates, totalling 12 fits


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.


    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00492922
    [200]	valid's binary_logloss: 0.00416918
    [300]	valid's binary_logloss: 0.00389234
    [400]	valid's binary_logloss: 0.00373652
    [500]	valid's binary_logloss: 0.0036475
    [600]	valid's binary_logloss: 0.00359889
    [700]	valid's binary_logloss: 0.00356482
    [800]	valid's binary_logloss: 0.0035412
    [900]	valid's binary_logloss: 0.00352275
    Early stopping, best iteration is:
    [927]	valid's binary_logloss: 0.00351921
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00500393
    [200]	valid's binary_logloss: 0.0042256
    [300]	valid's binary_logloss: 0.00395023
    [400]	valid's binary_logloss: 0.00381596
    [500]	valid's binary_logloss: 0.00373379
    [600]	valid's binary_logloss: 0.00368348
    [700]	valid's binary_logloss: 0.00364413
    [800]	valid's binary_logloss: 0.0036197
    [900]	valid's binary_logloss: 0.00359164
    [1000]	valid's binary_logloss: 0.00356601
    [1100]	valid's binary_logloss: 0.00355557
    [1200]	valid's binary_logloss: 0.00354476
    [1300]	valid's binary_logloss: 0.00353346
    [1400]	valid's binary_logloss: 0.00352819
    Early stopping, best iteration is:
    [1371]	valid's binary_logloss: 0.00352781
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00492089
    [200]	valid's binary_logloss: 0.00414336
    [300]	valid's binary_logloss: 0.00387093
    [400]	valid's binary_logloss: 0.00373963
    [500]	valid's binary_logloss: 0.00364771
    [600]	valid's binary_logloss: 0.00358239
    [700]	valid's binary_logloss: 0.00353957
    [800]	valid's binary_logloss: 0.00350767
    [900]	valid's binary_logloss: 0.00347788
    [1000]	valid's binary_logloss: 0.00345701
    [1100]	valid's binary_logloss: 0.00344497
    Early stopping, best iteration is:
    [1154]	valid's binary_logloss: 0.00343696
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00499867
    [200]	valid's binary_logloss: 0.0042013
    [300]	valid's binary_logloss: 0.00390103
    [400]	valid's binary_logloss: 0.00373636
    [500]	valid's binary_logloss: 0.00364971
    [600]	valid's binary_logloss: 0.00358742
    [700]	valid's binary_logloss: 0.00356176
    [800]	valid's binary_logloss: 0.00354897
    Early stopping, best iteration is:
    [795]	valid's binary_logloss: 0.0035462
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00509935
    [200]	valid's binary_logloss: 0.00426953
    [300]	valid's binary_logloss: 0.0039831
    [400]	valid's binary_logloss: 0.00382911
    [500]	valid's binary_logloss: 0.00372757
    [600]	valid's binary_logloss: 0.00365412
    [700]	valid's binary_logloss: 0.00360757
    [800]	valid's binary_logloss: 0.00358653
    [900]	valid's binary_logloss: 0.00357027
    Early stopping, best iteration is:
    [966]	valid's binary_logloss: 0.00356775
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00498213
    [200]	valid's binary_logloss: 0.00417594
    [300]	valid's binary_logloss: 0.00387595
    [400]	valid's binary_logloss: 0.00371738
    [500]	valid's binary_logloss: 0.00361393
    [600]	valid's binary_logloss: 0.00355529
    [700]	valid's binary_logloss: 0.00351279
    [800]	valid's binary_logloss: 0.00348081
    [900]	valid's binary_logloss: 0.0034685
    Early stopping, best iteration is:
    [941]	valid's binary_logloss: 0.00345943
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.0060526
    [200]	valid's binary_logloss: 0.00483909
    [300]	valid's binary_logloss: 0.00432076
    [400]	valid's binary_logloss: 0.00406284
    [500]	valid's binary_logloss: 0.00391959
    [600]	valid's binary_logloss: 0.00385545
    [700]	valid's binary_logloss: 0.0038168
    [800]	valid's binary_logloss: 0.00378549
    [900]	valid's binary_logloss: 0.00377395
    Early stopping, best iteration is:
    [930]	valid's binary_logloss: 0.00376985
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00620207
    [200]	valid's binary_logloss: 0.00494388
    [300]	valid's binary_logloss: 0.00439372
    [400]	valid's binary_logloss: 0.00409846
    [500]	valid's binary_logloss: 0.00393944
    [600]	valid's binary_logloss: 0.00384026
    [700]	valid's binary_logloss: 0.00378288
    [800]	valid's binary_logloss: 0.00376039
    [900]	valid's binary_logloss: 0.00373719
    Early stopping, best iteration is:
    [961]	valid's binary_logloss: 0.00373277
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00608733
    [200]	valid's binary_logloss: 0.00481942
    [300]	valid's binary_logloss: 0.0042904
    [400]	valid's binary_logloss: 0.00402943
    [500]	valid's binary_logloss: 0.00388441
    [600]	valid's binary_logloss: 0.00379042
    [700]	valid's binary_logloss: 0.00374198
    [800]	valid's binary_logloss: 0.0037067
    [900]	valid's binary_logloss: 0.00367681
    [1000]	valid's binary_logloss: 0.00365453
    Early stopping, best iteration is:
    [1016]	valid's binary_logloss: 0.0036491
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.0072638
    [200]	valid's binary_logloss: 0.00549575
    [300]	valid's binary_logloss: 0.00476488
    [400]	valid's binary_logloss: 0.00443187
    [500]	valid's binary_logloss: 0.00423277
    [600]	valid's binary_logloss: 0.00411464
    [700]	valid's binary_logloss: 0.00404472
    [800]	valid's binary_logloss: 0.00400516
    [900]	valid's binary_logloss: 0.00398859
    Early stopping, best iteration is:
    [941]	valid's binary_logloss: 0.00398137
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00758748
    [200]	valid's binary_logloss: 0.00567327
    [300]	valid's binary_logloss: 0.00492706
    [400]	valid's binary_logloss: 0.00454516
    [500]	valid's binary_logloss: 0.00432883
    [600]	valid's binary_logloss: 0.00418749
    [700]	valid's binary_logloss: 0.00411667
    [800]	valid's binary_logloss: 0.0040634
    [900]	valid's binary_logloss: 0.00403749
    Early stopping, best iteration is:
    [925]	valid's binary_logloss: 0.00402844
    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00742066
    [200]	valid's binary_logloss: 0.00546049
    [300]	valid's binary_logloss: 0.00472158
    [400]	valid's binary_logloss: 0.00437011
    [500]	valid's binary_logloss: 0.00417822
    [600]	valid's binary_logloss: 0.00403826
    [700]	valid's binary_logloss: 0.00397108
    [800]	valid's binary_logloss: 0.00391554
    [900]	valid's binary_logloss: 0.00388631
    [1000]	valid's binary_logloss: 0.00386949
    Early stopping, best iteration is:
    [1052]	valid's binary_logloss: 0.00385859


    [Parallel(n_jobs=1)]: Done  12 out of  12 | elapsed: 29.3min finished


    Training until validation scores don't improve for 30 rounds.
    [100]	valid's binary_logloss: 0.00468361
    [200]	valid's binary_logloss: 0.00392183
    [300]	valid's binary_logloss: 0.00364829
    [400]	valid's binary_logloss: 0.00348944
    [500]	valid's binary_logloss: 0.00337939
    [600]	valid's binary_logloss: 0.00330502
    [700]	valid's binary_logloss: 0.00325775
    [800]	valid's binary_logloss: 0.00322144
    [900]	valid's binary_logloss: 0.00319789
    [1000]	valid's binary_logloss: 0.0031711
    [1100]	valid's binary_logloss: 0.00315203
    Early stopping, best iteration is:
    [1157]	valid's binary_logloss: 0.00314025
    Best score reached: 0.9988364249236769 with params: {'scale_pos_weight': 1} 


#### Hyperparameters Tunning results 


```python



models=[i for i in tuned_model_scores]
f1_tunned=[tuned_model_scores[i]for i in tuned_model_scores ]
f1_bare=[Bare_model_scores[i[:-1]][-1] if i[:-1]=="lgbm"  else  Bare_model_scores[i][-1] for i in tuned_model_scores ]



fig = go.Figure()

fig.add_trace(go.Bar(
    x=models,
    y=f1_tunned,
    name='New F1 score',
    marker_color='LightSkyBlue'
))

fig.add_trace(go.Bar(
    x=models,
    y=f1_bare,
    name='Old F1 score  ',
    marker_color='lightsalmon'
))

fig.update_layout(barmode='group',yaxis=dict(range=[0.9,1]), xaxis_tickangle=-45, title="Hyperparameters Tunning's effect on F1 Score ",
    xaxis_title="Algorithms",
    yaxis_title="Scores",
    font=dict(
        family="Courier New, monospace",
        size=14,
        color="#7f7f7f"
    ))
fig.show()

```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="954f5e24-9f5b-4bce-b717-065ab1b0ad8c" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("954f5e24-9f5b-4bce-b717-065ab1b0ad8c")) {
                    Plotly.newPlot(
                        '954f5e24-9f5b-4bce-b717-065ab1b0ad8c',
                        [{"marker": {"color": "LightSkyBlue"}, "name": "New F1 score", "type": "bar", "x": ["logreg", "ridge", "svm", "knn", "rf", "xgb", "lgbm1", "lgbm2"], "y": [0.9697506390958892, 0.9034958723761228, 0.9712217256644158, 0.9969489906753376, 0.998462827182174, 0.9982498817867074, 0.9989020769061057, 0.9988364249236769]}, {"marker": {"color": "lightsalmon"}, "name": "Old F1 score  ", "type": "bar", "x": ["logreg", "ridge", "svm", "knn", "rf", "xgb", "lgbm1", "lgbm2"], "y": [0.9597506390958892, 0.8634958723761228, 0.9512217256644158, 0.9569489906753376, 0.9770589906753376, 0.9882498817867074, 0.9889020769061057, 0.9889020769061057]}],
                        {"barmode": "group", "font": {"color": "#7f7f7f", "family": "Courier New, monospace", "size": 14}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Hyperparameters Tunning's effect on F1 Score "}, "xaxis": {"tickangle": -45, "title": {"text": "Algorithms"}}, "yaxis": {"range": [0.9, 1], "title": {"text": "Scores"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('954f5e24-9f5b-4bce-b717-065ab1b0ad8c');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };

            </script>
        </div>
</body>
</html>


$\implies$ Based on the graph above we can conclude that the Hyperprameters tunning is a crucial step in every for every Machine learning  algorithm. Prevouisly using the baseline models we considred  only LGBM and XGB because they outperfomed  the rest of the algorithms. 
But when tunning the hyperpramters of all the algorithms we can see two major diffrences : 
 
 * The performance of all the models increased using the tunned hyperprameters vs the default ones

 * Some baseline models performed less than LGBM and XGB without tunning,  But after the Tunning they are back in the game , like(Knn and RandomForest) 

 * Still the tunning was done on sub sample of the data only 20% of the train data. KNN took so much time to tune and fit over the tunning subsample and more than 10 hours to predict the test set. So due to long time it needs to run, we won't be using KNN although it achieved good results on the tunning subsample and we will focus on LightGBM & XGBoost & RandomForest 


##Stacking


Ensemble learning helps improve machine learning results by combining several models. This approach allows the production of better predictive performance compared to a single model.

In order to further improve the performance of our models, We opted for using ensemble methods (Stacking) to combine our heterogeneous models.

`Stacking`: is an ensemble learning technique that combines multiple classification  models[LGMB,XGB,RF] via a meta-classifier [LGBM,XGB]. The base level models are trained based on a complete training set, then the meta-model is trained on the outputs of the base level model as features.

The base level  consists of 3 different clusters of learning algorithms[LGBM 5 models, XGBoost 2 models, RF 3 models] each cluster is a single model trained on a set of  different hyperpramaters. 

The choice of this architecture is based on Trial and Error, We tried a lot of architectures and a number of different algorithms.The picture below summarizes the best stacking algorithm I found. 


![image.png](data:image/png;base64,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xGPFdcQHBbEh4XKNgv6h7MS4TgVzsh8/9bj8kKo6LqU+YI8BGdKAO0iABEiABEiABEggjQk07fXH1E7jKXJqJEACJEACJEACKSZAxWOKTwCHj52ATvLi1tLr9covvzlcurq6rCoVFRVyy5UFQdtfu6gg4BqN45+fWxBQNGK7paUlqP4vvxncPtrjUF5eddEIqz+t9LT3X1xcLK2t/S7ZbuvifhIggewkcGCNP2nNsFkV2blArooESIAEEkxAW0AxOVeCwbI7EiABEiABEkgCAbpaJwEqu0wtgYoPt8jEXpGCggKpqqqS2of/mNbb41atSS0wjk4CJJBSAs8v2CD4sJAACZAACZAACZAACZAACZBAuhKYs3iS4BNrCVI8Tl5YI/iwkEAmE2hXCkfP0fNlsidfan5+t8ijz6T39mNPZjJuzp0ESIAESIAESIAESIAESIAESIAESCDLCUyYXyP4xFqCXK1P+n7smstYB2R9Ekg2gaZRtVL35GKpmTRRZP7lsumhn0vV5g/Sd3vxT5KNhP2TAAmQAAmQAAmQAAmQAAmQQFIIxGMBlZSJsFMSIIG0JMAYj2l5WjipgRI4NOUo6Xj5FWmuGW51le7bA10v25MACZAACZAACZAACZAACZBAKgjEYwGVinlyTBIggdQQYIzH1HDnqINAQCsd9VDpvj0ISDgECZAACZAACZAACZAACZAACZAACZAACQwaASoeBw01ByIBEiABEiABEiABEiABEiABEiABEiABEiCB3CFAxWPunGuulARIgARIgARIgARIgARIgARIgARIgARIgARiJrB07mrBJ9YSFONx5S1brPZMMhMrRtYnARIgARIgARIgARIgARIgARIgARIgARIggewk0LS3Pa6FBSkeP3ygzuqEise4WLIRCZAACZBAFhMYMbcii1fHpZEACZAACZAACZBAfAS0BdTFy2bH1wFbkQAJZDUBZrXO6tPLxZEACZAACSSKwNy7pyeqK/ZDAiRAAiRAAiRAAllDIF4rqKwBwIWQAAmEJcAYj2Hx8CAJkAAJkAAJkAAJkAAJkAAJkAAJkAAJkAAJkEA8BKh4jIca25AACZAACZAACZAACZAACZAACZAACZAACZAACYQlQFfrsHh4kARIgARIgARIgARIgARIwInA5IU1Tru5jwRIgAQSQmDm90YnpB92QgIkkFoCVDym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)

### Generating Base level output: 

The base level consists of 3 different clusters of learning algorithms[LGBM 5 models, XGBoost 2 models, RF 3 models] each cluster is a single model trained on a set of different hyperpramaters. 


`Set of different hyperpramaters` : this set is chosen based on the results of tuning the algorithm on a different parametric  search spaces.
you can find below a list of those set of hyperparmaters that we will use to generate the clusters of models of the base level of the stack.

$\implies$ We train each model on the X_train data while changing at each step the hyperprameters and saving the output of this model  , in order to use to create the output dataframe on which we will train our meta classifier.

#### Hyperprameters set for each cluster of models :


```python
params_lgbm=[
             {
                'colsample_bytree': 0.5236485492981339,
                'learning_rate': 0.1,
                'max_depth': -1,
                'min_child_samples': 184,
                'min_child_weight': 1e-05,
                'n_estimators': 2000,
                'num_leaves': 37,
                'reg_alpha': 0.1,
                'reg_lambda': 100,
                'subsample': 0.6814395442651335
             }, 
             {
                "eval_metric" : 'binary_logloss', 
                "eval_set" : [(X_base_test,y_base_test)],
                'objective': 'binary',
                'boosting': 'gbdt',
                'learning_rate': 0.01 ,
                'verbose': 3,
                'n_estimators' :2000,
                'min_child_samples': 50, 
                'num_leaves': 180, 
                'reg_alpha': 0.1, 
                'reg_lambda': 100, 
                "metric":"accuracy",
                'feature_fraction': 0.8, 
                'bagging_fraction': 0.5415639874077389,
                'bagging_freq': 1,
                'num_iteration':10000
            },
            {
                "eval_metric" : 'binary', 
                "eval_set" : [(X_base_test,y_base_test)],
                'eval_names': ['valid'],
                'objective': 'binary',
                'boosting': 'gbdt',
                'learning_rate': 0.01 ,
                'verbose': 3,
                'n_estimators' :2000,
                'colsample_bytree': 0.5236485492981339, 
                'min_child_samples': 184, 
                'min_child_weight': 1e-05, 
                'num_leaves': 69, 
                'reg_alpha': 0.1, 
                'reg_lambda': 100, 
                'subsample': 0.6814395442651335, 
                "metric":"accuracy",
                'feature_fraction': 0.8, 
                'bagging_fraction': 0.5415639874077389,
                'bagging_freq': 1,
                'min_child_samples': 50,
                'num_iteration':5000
                },
                {
                'lambda_l1': 0.0,
                'lambda_l2': 0.0,
                'num_leaves': 69,
                'feature_fraction': 0.8, 
                'bagging_fraction': 0.5415639874077389,
                'bagging_freq': 1,
                'min_child_samples': 50
                },
                {
                'colsample_bytree': 0.9501241488957805, 
                'min_child_samples': 301,
                'min_child_weight': 0.1,
                'num_leaves': 28,
                'reg_alpha': 0,
                'reg_lambda': 100,
                'subsample': 0.9326466073236168,
                'scale_pos_weight': 1} 
]
params_rf=[
           {'max_depth': 34,
             'min_samples_split': 14,
             'n_estimators':15},
            {'max_depth': 55,
             'min_samples_split': 7,
             'n_estimators':15},
            {'max_depth': 10,
             'min_samples_split': 30,
             'n_estimators':40
            }
           ]

params_XGB=[
             {'scale_pos_weight': 1, 
              'objective': 'binary:hinge',
              'min_depth': 6,
              'min_child_weight': 5,
              'eta': 0.1
              } , 
             {
              'scale_pos_weight': 2, 
              'objective': 'binary:hinge',
              'min_depth': 5, 
              'min_child_weight': 1,
              'eta': 0.3
              } 
]
```

We Loop through the hyperparamters list and generate & save the output of each model of each cluster in order to use it later on in creating the dataframe on which we fit and tune  our meta Classifier  

#### Output of LGBM Cluster : 

  * Test set (Testing Set to be submitted ) 

  * X_base_test ( Validation set ) 


```python
for i in range(len(params_lgbm)):
  opt_parameters=params_lgbm[i]
  start_time=time.time()
  clf_sw = lgb.LGBMClassifier()
  clf_sw.set_params(**opt_parameters)
  print("#################################################MODEL ",i )
  clf_sw.fit(X_base_train,y_base_train)
  print("Timming:",time.time()-start_time)
  prediction_test = clf_sw.predict(Test_scaled)
  print("Timming:",time.time()-start_time)
  np.savetxt('LGBM'+str(i)+'test.csv', prediction_test, fmt = '%1.0d', delimiter=',')


```

    #################################################MODEL  0
    Timming: 174.53604865074158
    Timming: 578.2241892814636
    #################################################MODEL  1


    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:118: UserWarning:
    
    Found `num_iteration` in params. Will use it instead of argument
    


    Timming: 1435.430725812912
    Timming: 10637.934380531311
    #################################################MODEL  2


    /usr/local/lib/python3.6/dist-packages/lightgbm/engine.py:118: UserWarning:
    
    Found `num_iteration` in params. Will use it instead of argument
    


    Timming: 617.9367344379425
    Timming: 2327.16743850708
    #################################################MODEL  3
    Timming: 12.772420883178711
    Timming: 26.347994327545166
    #################################################MODEL  4
    Timming: 12.358758687973022
    Timming: 24.0046546459198


#### Output of XGBoost Cluster 

  * Test set (Testing Set to be submitted ) 

  * X_base_test ( Validation set ) 


```python
for i in range(len(params_XGB)):
  opt_parameters=params_XGB[i]
  start_time=time.time()
  xg = xgb.XGBClassifier()
  xg.set_params(**opt_parameters)
  print("#################################################MODEL ",i )
  xg.fit(X_base_train,y_base_train)
  print("Timming:",time.time()-start_time)
  prediction_test = xg.predict(Test_scaled)
  print("Timming:",time.time()-start_time)
  np.savetxt('XGB'+str(i)+'test.csv', prediction_test, fmt = '%1.0d', delimiter=',')
```

    #################################################MODEL  0
    Timming: 109.83312106132507
    Timming: 119.3869001865387
    #################################################MODEL  1
    Timming: 113.97897553443909
    Timming: 122.81644821166992


#### Output of RandomForest Cluster:  

  * Test set (Testing Set to be submitted ) 

  * X_base_test ( Validation set ) 


```python
for i in range(len(params_rf)):
  opt_parameters=params_rf[i]
  start_time=time.time()
  clf_sw = RandomForestClassifier()
  clf_sw.set_params(**opt_parameters)
  print("#################################################MODEL ",i )
  clf_sw.fit(X_base_train,y_base_train)
  print("Timming:",time.time()-start_time)
  prediction_test = clf_sw.predict(Test_scaled)
  print("Timming:",time.time()-start_time)
  np.savetxt('RF'+str(i)+'test.csv', prediction_test, fmt = '%1.0d', delimiter=',')
```

    #################################################MODEL  0
    Timming: 148.21696734428406
    Timming: 156.8480086326599
    #################################################MODEL  1
    Timming: 152.24142813682556
    Timming: 160.79733610153198
    #################################################MODEL  2
    Timming: 118.85472559928894
    Timming: 128.14788818359375


###Creating the output DataFrames: 
  * Training & tuning dataframe : Concatination of prediction of X_base_train over all the models   
  
  * Testing Dataframe: Concatination of prediction of Test dataset (to be submitted ) over all the models . 

#### Creating  Training & Tuning  dataFrame : 

prediction of X_base_train data 



```python
# download the output of the base level models 
!rm -r ./stacking 
gdd.download_file_from_google_drive(file_id='1jiKfAt-w6MBfeYAQBSKzqyDrTU26NgXP',dest_path='./stack_data.zip')
!unzip ./stack_data.zip -d ./
```

    Archive:  ./stack_data.zip
      inflating: ./stacking/RF1.csv      
      inflating: ./stacking/RF0.csv      
      inflating: ./stacking/LGBM2.csv    
      inflating: ./stacking/LGBM4.csv    
      inflating: ./stacking/RF2.csv      
      inflating: ./stacking/LGBM3.csv    
      inflating: ./stacking/XGB0.csv     
      inflating: ./stacking/LGBM1.csv    
      inflating: ./stacking/LGBM0.csv    
      inflating: ./stacking/XGB1.csv     
      inflating: ./stacking/test/LGBM4test.csv  
      inflating: ./stacking/test/RF1test.csv  
      inflating: ./stacking/test/RF0test.csv  
      inflating: ./stacking/test/LGBM0test.csv  
      inflating: ./stacking/test/RF2test.csv  
      inflating: ./stacking/test/LGBM1test.csv  
      inflating: ./stacking/test/XGB0test.csv  
      inflating: ./stacking/test/LGBM2test.csv  
      inflating: ./stacking/test/XGB1test.csv  
      inflating: ./stacking/test/LGBM3test.csv  



```python
!ls ./stacking
```

    LGBM0.csv  LGBM2.csv  LGBM4.csv  RF1.csv  test	    XGB1.csv
    LGBM1.csv  LGBM3.csv  RF0.csv	 RF2.csv  XGB0.csv



```python

stacking_path="./stacking/"
stacking_df=pd.DataFrame(y_base_test).reset_index(drop=True)

import os 
for filename in os.listdir(stacking_path):
    model=filename.split('.')[0]
    if(model=="test"): 
      continue ## This is a directory not a data file!

    stacking_df[model]= pd.read_csv(stacking_path+filename,header=None).reset_index(drop=True)
    pred_positive=stacking_df[stacking_df[model]==1].count()[model]
    real_positive=stacking_df[stacking_df['y']==1].count()['y']
    ratio=pred_positive/real_positive ## we can use this ratio as an indiction of how good is our model
                                      ## ( we can't prove we are doing good but we can see if a model is 
                                      ## not working correctly  for exemple predecting 10% of real  postive values ) 
    print(model,"=>", ratio , '{', pred_positive , ' vs ' ,real_positive , '}')
```

    LGBM2 => 0.9948254556688292 { 12881  vs  12948 }
    RF0 => 0.9912727834414582 { 12835  vs  12948 }
    LGBM1 => 0.9943620636391721 { 12875  vs  12948 }
    LGBM0 => 0.994902687673772 { 12882  vs  12948 }
    LGBM4 => 0.9943620636391721 { 12875  vs  12948 }
    RF2 => 0.9808464627741736 { 12700  vs  12948 }
    XGB1 => 0.9908866234167439 { 12830  vs  12948 }
    LGBM3 => 1.0041705282669138 { 13002  vs  12948 }
    RF1 => 0.9911955514365153 { 12834  vs  12948 }
    XGB0 => 0.990809391411801 { 12829  vs  12948 }



```python
stacking_df.head()
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
      <th>y</th>
      <th>LGBM2</th>
      <th>RF0</th>
      <th>LGBM1</th>
      <th>LGBM0</th>
      <th>LGBM4</th>
      <th>RF2</th>
      <th>XGB1</th>
      <th>LGBM3</th>
      <th>RF1</th>
      <th>XGB0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



#####Feature engineering
We added some features to our dataframe like: 

  * the ratio of postive prediction over all the predictions

  * Majorty Vote 

  * Minorty Vote ( we set this feature to 1 if at least one model predected   )


```python

stacking_df["Pos_ratio"]=stacking_df.drop('y',axis=1).sum(axis=1)/len(stacking_df.drop('y',axis=1).columns)
stacking_df["minorty_vote"]=stacking_df.drop('y',axis=1).max(axis=1)
stacking_df["majorty_vote"]=(round(stacking_df["Pos_ratio"]))
#stacking_df[stacking_df["max"]==1]["max"].count(),stacking_df[stacking_df["y"]==1]["y"].count()
#Test_stack=#### TOBEDONELATER
X_stack=stacking_df.drop('y',axis=1)
y_stack=stacking_df['y']
stacking_df.head()

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
      <th>y</th>
      <th>LGBM2</th>
      <th>RF0</th>
      <th>LGBM1</th>
      <th>LGBM0</th>
      <th>LGBM4</th>
      <th>RF2</th>
      <th>XGB1</th>
      <th>LGBM3</th>
      <th>RF1</th>
      <th>XGB0</th>
      <th>Pos_ratio</th>
      <th>minorty_vote</th>
      <th>majorty_vote</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



##### Splitting the data again for the  Meta-classifier models :



```python
X_stack_train , X_stack_test, y_stack_train, y_stack_test= train_test_split(X_stack, y_stack, test_size=0.5,random_state =666)

```

#### Creating  Test dataFrame : 

prediction of Test.csv data 



```python

stacking_test_path="./stacking/test/"


filename=os.listdir(stacking_test_path)[0]
stacking_test_df=pd.read_csv(stacking_test_path+filename,header=None,names=[filename.split('.')[0][:-4]]).reset_index(drop=True)

import os 
for filename in os.listdir(stacking_test_path)[1:]:
    model=filename.split('.')[0]
   
    stacking_test_df[model[:-4]]= pd.read_csv(stacking_test_path+filename,header=None).reset_index(drop=True)
    print(model)


stacking_test_df["Pos_ratio"]=stacking_test_df.sum(axis=1)/len(stacking_test_df.columns)
stacking_test_df["minorty_vote"]=stacking_test_df.max(axis=1)
stacking_test_df["majorty_vote"]=(round(stacking_test_df["Pos_ratio"]))

stacking_test_df.head()
```

    RF2test
    LGBM3test
    RF0test
    RF1test
    LGBM0test
    LGBM1test
    XGB0test
    XGB1test
    LGBM2test





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
      <th>LGBM4</th>
      <th>RF2</th>
      <th>LGBM3</th>
      <th>RF0</th>
      <th>RF1</th>
      <th>LGBM0</th>
      <th>LGBM1</th>
      <th>XGB0</th>
      <th>XGB1</th>
      <th>LGBM2</th>
      <th>Pos_ratio</th>
      <th>minorty_vote</th>
      <th>majorty_vote</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



###Meta classifier :  

As Meta classifiers we Use LGBM and XGB as showen in picture above.

* We will tune the models on the output of the base level (X_stack_train , y_stack_train)

* predict (stacking_test_df) 

#### LGBM

##### Meta classifier LGBM hyperparameter Tunning: 


```python
start_time=time.time()
fit_params = { 
            "early_stopping_rounds" : 10, 
             "eval_metric" : 'binary', 
             "eval_set" : [(X_stack_test,y_stack_test)],
             'eval_names': ['valid'],
             'verbose': 1}

param_test = { 
              'learning_rate' : [0.001,0.01,0.05, 0.1, 0.3,1],
              'n_estimators' : [100, 200,  400,  600, 800, 1000, 2000],
              'num_leaves': randint(6, 50), 
              'min_child_samples': randint(50, 300), 
              'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
              'subsample': uniform(loc=0.2, scale=0.8), 
              'max_depth': [-1, 1, 2, 3, 4, 5, 6, 7],
              'colsample_bytree': uniform(loc=0.4, scale=0.6),
              'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
              'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}

#number of combinations
n_iter = 200 

#intialize lgbm and lunch the search
lgbm_clf = lgb.LGBMClassifier(random_state=666, silent=False, metric='None', n_jobs=-1)
grid_search = RandomizedSearchCV(
    estimator=lgbm_clf, param_distributions=param_test, 
    n_iter=n_iter,
    scoring='accuracy',
    cv=3,
    refit=True,
    random_state=666,
    verbose=True)
# we tune out model on The output of the base level
grid_search.fit(X_stack_train, y_stack_train, **fit_params)
print('Best score reached: {} with params: {} '.format(grid_search.best_score_, grid_search.best_params_))

opt_parameters =  grid_search.best_params_
print("-------------------Timing: {} ------------------".format(time.time()-start_time))
```

    Fitting 3 folds for each of 200 candidates, totalling 600 fits
    [1]	valid's binary_logloss: 0.0251396
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.0251396


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.


    [1;30;43mStreaming output truncated to the last 5000 lines.[0m
    [10]	valid's binary_logloss: 0.0265165
    [11]	valid's binary_logloss: 0.0265165
    Early stopping, best iteration is:
    [1]	valid's binary_logloss: 0.0265165
    [1]	valid's binary_logloss: 0.0265123
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.0265123
    [3]	valid's binary_logloss: 0.0265123
    [4]	valid's binary_logloss: 0.0265123
    [5]	valid's binary_logloss: 0.0265123
    [6]	valid's binary_logloss: 0.0265123
    [7]	valid's binary_logloss: 0.0265123
    [8]	valid's binary_logloss: 0.0265123
    [9]	valid's binary_logloss: 0.0265123
    [10]	valid's binary_logloss: 0.0265123
    [11]	valid's binary_logloss: 0.0265123
    Early stopping, best iteration is:
    [1]	valid's binary_logloss: 0.0265123
    [1]	valid's binary_logloss: 0.147164
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.139742
    [3]	valid's binary_logloss: 0.13365
    [4]	valid's binary_logloss: 0.128461
    [5]	valid's binary_logloss: 0.123933
    [6]	valid's binary_logloss: 0.119904
    [7]	valid's binary_logloss: 0.116275
    [8]	valid's binary_logloss: 0.112969
    [9]	valid's binary_logloss: 0.109928
    [10]	valid's binary_logloss: 0.107116
    [11]	valid's binary_logloss: 0.104496
    [12]	valid's binary_logloss: 0.102045
    [13]	valid's binary_logloss: 0.0997401
    [14]	valid's binary_logloss: 0.0975638
    [15]	valid's binary_logloss: 0.0955043
    [16]	valid's binary_logloss: 0.0935471
    [17]	valid's binary_logloss: 0.0916834
    [18]	valid's binary_logloss: 0.0899046
    [19]	valid's binary_logloss: 0.0882037
    [20]	valid's binary_logloss: 0.0865727
    [21]	valid's binary_logloss: 0.0850075
    [22]	valid's binary_logloss: 0.0835017
    [23]	valid's binary_logloss: 0.0820514
    [24]	valid's binary_logloss: 0.0806534
    [25]	valid's binary_logloss: 0.0793059
    [26]	valid's binary_logloss: 0.0779991
    [27]	valid's binary_logloss: 0.0767346
    [28]	valid's binary_logloss: 0.0755112
    [29]	valid's binary_logloss: 0.0743242
    [30]	valid's binary_logloss: 0.0731718
    [31]	valid's binary_logloss: 0.0720534
    [32]	valid's binary_logloss: 0.0709656
    [33]	valid's binary_logloss: 0.0699112
    [34]	valid's binary_logloss: 0.0688825
    [35]	valid's binary_logloss: 0.0678796
    [36]	valid's binary_logloss: 0.0669031
    [37]	valid's binary_logloss: 0.0659504
    [38]	valid's binary_logloss: 0.0650216
    [39]	valid's binary_logloss: 0.0641151
    [40]	valid's binary_logloss: 0.0632299
    [41]	valid's binary_logloss: 0.062365
    [42]	valid's binary_logloss: 0.0615203
    [43]	valid's binary_logloss: 0.060694
    [44]	valid's binary_logloss: 0.0598857
    [45]	valid's binary_logloss: 0.0590951
    [46]	valid's binary_logloss: 0.0583211
    [47]	valid's binary_logloss: 0.0575633
    [48]	valid's binary_logloss: 0.0568212
    [49]	valid's binary_logloss: 0.0560941
    [50]	valid's binary_logloss: 0.0553816
    [51]	valid's binary_logloss: 0.0546832
    [52]	valid's binary_logloss: 0.0539983
    [53]	valid's binary_logloss: 0.0533267
    [54]	valid's binary_logloss: 0.0526681
    [55]	valid's binary_logloss: 0.0520215
    [56]	valid's binary_logloss: 0.0513869
    [57]	valid's binary_logloss: 0.050764
    [58]	valid's binary_logloss: 0.0501522
    [59]	valid's binary_logloss: 0.0495534
    [60]	valid's binary_logloss: 0.0489633
    [61]	valid's binary_logloss: 0.0483836
    [62]	valid's binary_logloss: 0.047814
    [63]	valid's binary_logloss: 0.0472541
    [64]	valid's binary_logloss: 0.0467038
    [65]	valid's binary_logloss: 0.046163
    [66]	valid's binary_logloss: 0.045631
    [67]	valid's binary_logloss: 0.0451078
    [68]	valid's binary_logloss: 0.0445931
    [69]	valid's binary_logloss: 0.0440868
    [70]	valid's binary_logloss: 0.0435888
    [71]	valid's binary_logloss: 0.0430986
    [72]	valid's binary_logloss: 0.0426163
    [73]	valid's binary_logloss: 0.0421416
    [74]	valid's binary_logloss: 0.0416742
    [75]	valid's binary_logloss: 0.0412142
    [76]	valid's binary_logloss: 0.0407612
    [77]	valid's binary_logloss: 0.0403152
    [78]	valid's binary_logloss: 0.039876
    [79]	valid's binary_logloss: 0.0394434
    [80]	valid's binary_logloss: 0.0390172
    [81]	valid's binary_logloss: 0.0385976
    [82]	valid's binary_logloss: 0.038184
    [83]	valid's binary_logloss: 0.0377766
    [84]	valid's binary_logloss: 0.0373752
    [85]	valid's binary_logloss: 0.0369798
    [86]	valid's binary_logloss: 0.03659
    [87]	valid's binary_logloss: 0.0362057
    [88]	valid's binary_logloss: 0.0358269
    [89]	valid's binary_logloss: 0.0354536
    [90]	valid's binary_logloss: 0.0350854
    [91]	valid's binary_logloss: 0.0347226
    [92]	valid's binary_logloss: 0.0343647
    [93]	valid's binary_logloss: 0.0340119
    [94]	valid's binary_logloss: 0.0336639
    [95]	valid's binary_logloss: 0.0333223
    [96]	valid's binary_logloss: 0.0329838
    [97]	valid's binary_logloss: 0.03265
    [98]	valid's binary_logloss: 0.0323207
    [99]	valid's binary_logloss: 0.0319975
    [100]	valid's binary_logloss: 0.0316771
    [101]	valid's binary_logloss: 0.031361
    [102]	valid's binary_logloss: 0.0310491
    [103]	valid's binary_logloss: 0.030743
    [104]	valid's binary_logloss: 0.0304393
    [105]	valid's binary_logloss: 0.0301397
    [106]	valid's binary_logloss: 0.0298441
    [107]	valid's binary_logloss: 0.0295523
    [108]	valid's binary_logloss: 0.0292644
    [109]	valid's binary_logloss: 0.0289802
    [110]	valid's binary_logloss: 0.0286997
    [111]	valid's binary_logloss: 0.0284229
    [112]	valid's binary_logloss: 0.0281496
    [113]	valid's binary_logloss: 0.0278799
    [114]	valid's binary_logloss: 0.0276136
    [115]	valid's binary_logloss: 0.0273508
    [116]	valid's binary_logloss: 0.0270913
    [117]	valid's binary_logloss: 0.0268352
    [118]	valid's binary_logloss: 0.0265823
    [119]	valid's binary_logloss: 0.0263325
    [120]	valid's binary_logloss: 0.0260859
    [121]	valid's binary_logloss: 0.0258425
    [122]	valid's binary_logloss: 0.025602
    [123]	valid's binary_logloss: 0.0253647
    [124]	valid's binary_logloss: 0.0251302
    [125]	valid's binary_logloss: 0.0248987
    [126]	valid's binary_logloss: 0.0246701
    [127]	valid's binary_logloss: 0.0244444
    [128]	valid's binary_logloss: 0.0242214
    [129]	valid's binary_logloss: 0.0240012
    [130]	valid's binary_logloss: 0.0237837
    [131]	valid's binary_logloss: 0.0235702
    [132]	valid's binary_logloss: 0.023358
    [133]	valid's binary_logloss: 0.0231483
    [134]	valid's binary_logloss: 0.0229413
    [135]	valid's binary_logloss: 0.0227368
    [136]	valid's binary_logloss: 0.0225347
    [137]	valid's binary_logloss: 0.0223351
    [138]	valid's binary_logloss: 0.0221379
    [139]	valid's binary_logloss: 0.0219431
    [140]	valid's binary_logloss: 0.0217507
    [141]	valid's binary_logloss: 0.0215605
    [142]	valid's binary_logloss: 0.0213726
    [143]	valid's binary_logloss: 0.021187
    [144]	valid's binary_logloss: 0.021004
    [145]	valid's binary_logloss: 0.0208228
    [146]	valid's binary_logloss: 0.0206438
    [147]	valid's binary_logloss: 0.020467
    [148]	valid's binary_logloss: 0.0202922
    [149]	valid's binary_logloss: 0.0201196
    [150]	valid's binary_logloss: 0.019949
    [151]	valid's binary_logloss: 0.0197804
    [152]	valid's binary_logloss: 0.0196138
    [153]	valid's binary_logloss: 0.0194492
    [154]	valid's binary_logloss: 0.0192866
    [155]	valid's binary_logloss: 0.0191257
    [156]	valid's binary_logloss: 0.0189669
    [157]	valid's binary_logloss: 0.0188099
    [158]	valid's binary_logloss: 0.0186547
    [159]	valid's binary_logloss: 0.0185013
    [160]	valid's binary_logloss: 0.0183496
    [161]	valid's binary_logloss: 0.0181999
    [162]	valid's binary_logloss: 0.0180519
    [163]	valid's binary_logloss: 0.0179056
    [164]	valid's binary_logloss: 0.017761
    [165]	valid's binary_logloss: 0.0176198
    [166]	valid's binary_logloss: 0.0174785
    [167]	valid's binary_logloss: 0.0173389
    [168]	valid's binary_logloss: 0.017201
    [169]	valid's binary_logloss: 0.0170646
    [170]	valid's binary_logloss: 0.0169298
    [171]	valid's binary_logloss: 0.0167966
    [172]	valid's binary_logloss: 0.0166648
    [173]	valid's binary_logloss: 0.0165361
    [174]	valid's binary_logloss: 0.0164072
    [175]	valid's binary_logloss: 0.01628
    [176]	valid's binary_logloss: 0.0161543
    [177]	valid's binary_logloss: 0.0160298
    [178]	valid's binary_logloss: 0.015907
    [179]	valid's binary_logloss: 0.0157854
    [180]	valid's binary_logloss: 0.0156654
    [181]	valid's binary_logloss: 0.0155466
    [182]	valid's binary_logloss: 0.0154291
    [183]	valid's binary_logloss: 0.0153133
    [184]	valid's binary_logloss: 0.0151987
    [185]	valid's binary_logloss: 0.0150853
    [186]	valid's binary_logloss: 0.0149746
    [187]	valid's binary_logloss: 0.0148638
    [188]	valid's binary_logloss: 0.0147542
    [189]	valid's binary_logloss: 0.014646
    [190]	valid's binary_logloss: 0.0145389
    [191]	valid's binary_logloss: 0.014433
    [192]	valid's binary_logloss: 0.0143283
    [193]	valid's binary_logloss: 0.0142248
    [194]	valid's binary_logloss: 0.0141225
    [195]	valid's binary_logloss: 0.0140213
    [196]	valid's binary_logloss: 0.0139213
    [197]	valid's binary_logloss: 0.0138225
    [198]	valid's binary_logloss: 0.0137248
    [199]	valid's binary_logloss: 0.0136286
    [200]	valid's binary_logloss: 0.013533
    Did not meet early stopping. Best iteration is:
    [200]	valid's binary_logloss: 0.013533
    [1]	valid's binary_logloss: 0.147145
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.139715
    [3]	valid's binary_logloss: 0.133615
    [4]	valid's binary_logloss: 0.128422
    [5]	valid's binary_logloss: 0.12389
    [6]	valid's binary_logloss: 0.11986
    [7]	valid's binary_logloss: 0.116228
    [8]	valid's binary_logloss: 0.11292
    [9]	valid's binary_logloss: 0.10988
    [10]	valid's binary_logloss: 0.107067
    [11]	valid's binary_logloss: 0.104447
    [12]	valid's binary_logloss: 0.101995
    [13]	valid's binary_logloss: 0.0996897
    [14]	valid's binary_logloss: 0.0975143
    [15]	valid's binary_logloss: 0.0954524
    [16]	valid's binary_logloss: 0.093496
    [17]	valid's binary_logloss: 0.091633
    [18]	valid's binary_logloss: 0.0898547
    [19]	valid's binary_logloss: 0.0881538
    [20]	valid's binary_logloss: 0.0865236
    [21]	valid's binary_logloss: 0.0849568
    [22]	valid's binary_logloss: 0.0834502
    [23]	valid's binary_logloss: 0.0819994
    [24]	valid's binary_logloss: 0.0806014
    [25]	valid's binary_logloss: 0.0792541
    [26]	valid's binary_logloss: 0.077948
    [27]	valid's binary_logloss: 0.0766841
    [28]	valid's binary_logloss: 0.0754609
    [29]	valid's binary_logloss: 0.0742738
    [30]	valid's binary_logloss: 0.0731218
    [31]	valid's binary_logloss: 0.0720041
    [32]	valid's binary_logloss: 0.0709168
    [33]	valid's binary_logloss: 0.0698626
    [34]	valid's binary_logloss: 0.0688332
    [35]	valid's binary_logloss: 0.0678303
    [36]	valid's binary_logloss: 0.0668538
    [37]	valid's binary_logloss: 0.0659011
    [38]	valid's binary_logloss: 0.064973
    [39]	valid's binary_logloss: 0.064066
    [40]	valid's binary_logloss: 0.0631805
    [41]	valid's binary_logloss: 0.0623164
    [42]	valid's binary_logloss: 0.0614711
    [43]	valid's binary_logloss: 0.0606456
    [44]	valid's binary_logloss: 0.0598372
    [45]	valid's binary_logloss: 0.0590472
    [46]	valid's binary_logloss: 0.058273
    [47]	valid's binary_logloss: 0.057516
    [48]	valid's binary_logloss: 0.0567737
    [49]	valid's binary_logloss: 0.056047
    [50]	valid's binary_logloss: 0.0553348
    [51]	valid's binary_logloss: 0.0546362
    [52]	valid's binary_logloss: 0.0539517
    [53]	valid's binary_logloss: 0.0532803
    [54]	valid's binary_logloss: 0.0526216
    [55]	valid's binary_logloss: 0.0519753
    [56]	valid's binary_logloss: 0.0513411
    [57]	valid's binary_logloss: 0.0507182
    [58]	valid's binary_logloss: 0.0501063
    [59]	valid's binary_logloss: 0.0495079
    [60]	valid's binary_logloss: 0.0489176
    [61]	valid's binary_logloss: 0.0483378
    [62]	valid's binary_logloss: 0.0477687
    [63]	valid's binary_logloss: 0.0472091
    [64]	valid's binary_logloss: 0.0466589
    [65]	valid's binary_logloss: 0.0461185
    [66]	valid's binary_logloss: 0.0455869
    [67]	valid's binary_logloss: 0.0450636
    [68]	valid's binary_logloss: 0.0445489
    [69]	valid's binary_logloss: 0.0440429
    [70]	valid's binary_logloss: 0.0435448
    [71]	valid's binary_logloss: 0.0430549
    [72]	valid's binary_logloss: 0.0425729
    [73]	valid's binary_logloss: 0.0420981
    [74]	valid's binary_logloss: 0.0416309
    [75]	valid's binary_logloss: 0.0411712
    [76]	valid's binary_logloss: 0.0407182
    [77]	valid's binary_logloss: 0.0402725
    [78]	valid's binary_logloss: 0.0398336
    [79]	valid's binary_logloss: 0.0394011
    [80]	valid's binary_logloss: 0.038975
    [81]	valid's binary_logloss: 0.0385556
    [82]	valid's binary_logloss: 0.0381421
    [83]	valid's binary_logloss: 0.0377346
    [84]	valid's binary_logloss: 0.0373334
    [85]	valid's binary_logloss: 0.0369379
    [86]	valid's binary_logloss: 0.0365481
    [87]	valid's binary_logloss: 0.0361644
    [88]	valid's binary_logloss: 0.0357856
    [89]	valid's binary_logloss: 0.0354126
    [90]	valid's binary_logloss: 0.0350446
    [91]	valid's binary_logloss: 0.0346818
    [92]	valid's binary_logloss: 0.0343243
    [93]	valid's binary_logloss: 0.0339716
    [94]	valid's binary_logloss: 0.033624
    [95]	valid's binary_logloss: 0.0332831
    [96]	valid's binary_logloss: 0.0329448
    [97]	valid's binary_logloss: 0.0326111
    [98]	valid's binary_logloss: 0.0322822
    [99]	valid's binary_logloss: 0.0319595
    [100]	valid's binary_logloss: 0.0316393
    [101]	valid's binary_logloss: 0.0313233
    [102]	valid's binary_logloss: 0.0310116
    [103]	valid's binary_logloss: 0.030706
    [104]	valid's binary_logloss: 0.0304025
    [105]	valid's binary_logloss: 0.0301031
    [106]	valid's binary_logloss: 0.0298077
    [107]	valid's binary_logloss: 0.0295162
    [108]	valid's binary_logloss: 0.0292286
    [109]	valid's binary_logloss: 0.0289444
    [110]	valid's binary_logloss: 0.0286642
    [111]	valid's binary_logloss: 0.0283877
    [112]	valid's binary_logloss: 0.0281147
    [113]	valid's binary_logloss: 0.0278451
    [114]	valid's binary_logloss: 0.0275791
    [115]	valid's binary_logloss: 0.0273166
    [116]	valid's binary_logloss: 0.0270574
    [117]	valid's binary_logloss: 0.0268016
    [118]	valid's binary_logloss: 0.0265486
    [119]	valid's binary_logloss: 0.0262992
    [120]	valid's binary_logloss: 0.0260531
    [121]	valid's binary_logloss: 0.0258099
    [122]	valid's binary_logloss: 0.0255699
    [123]	valid's binary_logloss: 0.0253328
    [124]	valid's binary_logloss: 0.0250987
    [125]	valid's binary_logloss: 0.0248674
    [126]	valid's binary_logloss: 0.0246391
    [127]	valid's binary_logloss: 0.0244136
    [128]	valid's binary_logloss: 0.0241909
    [129]	valid's binary_logloss: 0.0239709
    [130]	valid's binary_logloss: 0.0237537
    [131]	valid's binary_logloss: 0.0235407
    [132]	valid's binary_logloss: 0.0233287
    [133]	valid's binary_logloss: 0.0231194
    [134]	valid's binary_logloss: 0.0229126
    [135]	valid's binary_logloss: 0.0227083
    [136]	valid's binary_logloss: 0.0225066
    [137]	valid's binary_logloss: 0.0223072
    [138]	valid's binary_logloss: 0.0221103
    [139]	valid's binary_logloss: 0.0219158
    [140]	valid's binary_logloss: 0.0217236
    [141]	valid's binary_logloss: 0.0215337
    [142]	valid's binary_logloss: 0.0213461
    [143]	valid's binary_logloss: 0.0211608
    [144]	valid's binary_logloss: 0.0209776
    [145]	valid's binary_logloss: 0.0207968
    [146]	valid's binary_logloss: 0.0206181
    [147]	valid's binary_logloss: 0.0204412
    [148]	valid's binary_logloss: 0.0202668
    [149]	valid's binary_logloss: 0.0200943
    [150]	valid's binary_logloss: 0.0199239
    [151]	valid's binary_logloss: 0.0197553
    [152]	valid's binary_logloss: 0.0195887
    [153]	valid's binary_logloss: 0.0194244
    [154]	valid's binary_logloss: 0.0192618
    [155]	valid's binary_logloss: 0.0191013
    [156]	valid's binary_logloss: 0.0189427
    [157]	valid's binary_logloss: 0.0187859
    [158]	valid's binary_logloss: 0.0186308
    [159]	valid's binary_logloss: 0.018478
    [160]	valid's binary_logloss: 0.018327
    [161]	valid's binary_logloss: 0.0181774
    [162]	valid's binary_logloss: 0.0180295
    [163]	valid's binary_logloss: 0.0178832
    [164]	valid's binary_logloss: 0.0177386
    [165]	valid's binary_logloss: 0.0175974
    [166]	valid's binary_logloss: 0.0174564
    [167]	valid's binary_logloss: 0.017317
    [168]	valid's binary_logloss: 0.017179
    [169]	valid's binary_logloss: 0.0170428
    [170]	valid's binary_logloss: 0.0169083
    [171]	valid's binary_logloss: 0.0167753
    [172]	valid's binary_logloss: 0.0166438
    [173]	valid's binary_logloss: 0.0165153
    [174]	valid's binary_logloss: 0.0163866
    [175]	valid's binary_logloss: 0.0162596
    [176]	valid's binary_logloss: 0.0161342
    [177]	valid's binary_logloss: 0.0160101
    [178]	valid's binary_logloss: 0.0158875
    [179]	valid's binary_logloss: 0.0157663
    [180]	valid's binary_logloss: 0.0156462
    [181]	valid's binary_logloss: 0.0155277
    [182]	valid's binary_logloss: 0.0154107
    [183]	valid's binary_logloss: 0.015295
    [184]	valid's binary_logloss: 0.0151803
    [185]	valid's binary_logloss: 0.0150672
    [186]	valid's binary_logloss: 0.0149568
    [187]	valid's binary_logloss: 0.014846
    [188]	valid's binary_logloss: 0.0147365
    [189]	valid's binary_logloss: 0.0146285
    [190]	valid's binary_logloss: 0.0145214
    [191]	valid's binary_logloss: 0.0144159
    [192]	valid's binary_logloss: 0.0143113
    [193]	valid's binary_logloss: 0.0142078
    [194]	valid's binary_logloss: 0.0141055
    [195]	valid's binary_logloss: 0.0140044
    [196]	valid's binary_logloss: 0.0139045
    [197]	valid's binary_logloss: 0.0138059
    [198]	valid's binary_logloss: 0.0137082
    [199]	valid's binary_logloss: 0.0136117
    [200]	valid's binary_logloss: 0.0135164
    Did not meet early stopping. Best iteration is:
    [200]	valid's binary_logloss: 0.0135164
    [1]	valid's binary_logloss: 0.147155
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.13973
    [3]	valid's binary_logloss: 0.133634
    [4]	valid's binary_logloss: 0.128445
    [5]	valid's binary_logloss: 0.123916
    [6]	valid's binary_logloss: 0.119889
    [7]	valid's binary_logloss: 0.116259
    [8]	valid's binary_logloss: 0.112947
    [9]	valid's binary_logloss: 0.109907
    [10]	valid's binary_logloss: 0.107095
    [11]	valid's binary_logloss: 0.104475
    [12]	valid's binary_logloss: 0.102024
    [13]	valid's binary_logloss: 0.0997199
    [14]	valid's binary_logloss: 0.0975444
    [15]	valid's binary_logloss: 0.0954817
    [16]	valid's binary_logloss: 0.0935259
    [17]	valid's binary_logloss: 0.0916635
    [18]	valid's binary_logloss: 0.0898858
    [19]	valid's binary_logloss: 0.0881845
    [20]	valid's binary_logloss: 0.0865547
    [21]	valid's binary_logloss: 0.08499
    [22]	valid's binary_logloss: 0.0834843
    [23]	valid's binary_logloss: 0.0820351
    [24]	valid's binary_logloss: 0.0806367
    [25]	valid's binary_logloss: 0.0792892
    [26]	valid's binary_logloss: 0.0779825
    [27]	valid's binary_logloss: 0.0767181
    [28]	valid's binary_logloss: 0.075493
    [29]	valid's binary_logloss: 0.0743052
    [30]	valid's binary_logloss: 0.0731527
    [31]	valid's binary_logloss: 0.0720334
    [32]	valid's binary_logloss: 0.0709457
    [33]	valid's binary_logloss: 0.0698913
    [34]	valid's binary_logloss: 0.0688617
    [35]	valid's binary_logloss: 0.0678601
    [36]	valid's binary_logloss: 0.066883
    [37]	valid's binary_logloss: 0.0659305
    [38]	valid's binary_logloss: 0.0650013
    [39]	valid's binary_logloss: 0.0640954
    [40]	valid's binary_logloss: 0.06321
    [41]	valid's binary_logloss: 0.0623451
    [42]	valid's binary_logloss: 0.0614998
    [43]	valid's binary_logloss: 0.0606743
    [44]	valid's binary_logloss: 0.0598661
    [45]	valid's binary_logloss: 0.0590762
    [46]	valid's binary_logloss: 0.0583022
    [47]	valid's binary_logloss: 0.0575448
    [48]	valid's binary_logloss: 0.0568027
    [49]	valid's binary_logloss: 0.0560757
    [50]	valid's binary_logloss: 0.0553633
    [51]	valid's binary_logloss: 0.0546649
    [52]	valid's binary_logloss: 0.0539802
    [53]	valid's binary_logloss: 0.0533087
    [54]	valid's binary_logloss: 0.05265
    [55]	valid's binary_logloss: 0.0520037
    [56]	valid's binary_logloss: 0.0513693
    [57]	valid's binary_logloss: 0.0507466
    [58]	valid's binary_logloss: 0.0501353
    [59]	valid's binary_logloss: 0.0495366
    [60]	valid's binary_logloss: 0.0489468
    [61]	valid's binary_logloss: 0.0483673
    [62]	valid's binary_logloss: 0.0477978
    [63]	valid's binary_logloss: 0.0472382
    [64]	valid's binary_logloss: 0.0466884
    [65]	valid's binary_logloss: 0.0461475
    [66]	valid's binary_logloss: 0.0456161
    [67]	valid's binary_logloss: 0.0450932
    [68]	valid's binary_logloss: 0.0445786
    [69]	valid's binary_logloss: 0.0440724
    [70]	valid's binary_logloss: 0.0435747
    [71]	valid's binary_logloss: 0.0430845
    [72]	valid's binary_logloss: 0.0426023
    [73]	valid's binary_logloss: 0.0421279
    [74]	valid's binary_logloss: 0.0416606
    [75]	valid's binary_logloss: 0.0412007
    [76]	valid's binary_logloss: 0.0407481
    [77]	valid's binary_logloss: 0.040302
    [78]	valid's binary_logloss: 0.039863
    [79]	valid's binary_logloss: 0.0394304
    [80]	valid's binary_logloss: 0.0390046
    [81]	valid's binary_logloss: 0.038585
    [82]	valid's binary_logloss: 0.0381715
    [83]	valid's binary_logloss: 0.0377642
    [84]	valid's binary_logloss: 0.0373629
    [85]	valid's binary_logloss: 0.0369675
    [86]	valid's binary_logloss: 0.0365778
    [87]	valid's binary_logloss: 0.0361939
    [88]	valid's binary_logloss: 0.0358152
    [89]	valid's binary_logloss: 0.0354418
    [90]	valid's binary_logloss: 0.0350738
    [91]	valid's binary_logloss: 0.0347113
    [92]	valid's binary_logloss: 0.0343536
    [93]	valid's binary_logloss: 0.0340008
    [94]	valid's binary_logloss: 0.033653
    [95]	valid's binary_logloss: 0.0333116
    [96]	valid's binary_logloss: 0.0329734
    [97]	valid's binary_logloss: 0.0326397
    [98]	valid's binary_logloss: 0.0323106
    [99]	valid's binary_logloss: 0.0319877
    [100]	valid's binary_logloss: 0.0316673
    [101]	valid's binary_logloss: 0.0313514
    [102]	valid's binary_logloss: 0.0310397
    [103]	valid's binary_logloss: 0.0307339
    [104]	valid's binary_logloss: 0.0304304
    [105]	valid's binary_logloss: 0.030131
    [106]	valid's binary_logloss: 0.0298355
    [107]	valid's binary_logloss: 0.0295437
    [108]	valid's binary_logloss: 0.029256
    [109]	valid's binary_logloss: 0.0289718
    [110]	valid's binary_logloss: 0.0286914
    [111]	valid's binary_logloss: 0.0284147
    [112]	valid's binary_logloss: 0.0281415
    [113]	valid's binary_logloss: 0.0278719
    [114]	valid's binary_logloss: 0.0276057
    [115]	valid's binary_logloss: 0.0273431
    [116]	valid's binary_logloss: 0.0270837
    [117]	valid's binary_logloss: 0.0268277
    [118]	valid's binary_logloss: 0.026575
    [119]	valid's binary_logloss: 0.0263252
    [120]	valid's binary_logloss: 0.0260792
    [121]	valid's binary_logloss: 0.0258357
    [122]	valid's binary_logloss: 0.0255955
    [123]	valid's binary_logloss: 0.0253583
    [124]	valid's binary_logloss: 0.025124
    [125]	valid's binary_logloss: 0.0248925
    [126]	valid's binary_logloss: 0.024664
    [127]	valid's binary_logloss: 0.0244384
    [128]	valid's binary_logloss: 0.0242154
    [129]	valid's binary_logloss: 0.0239953
    [130]	valid's binary_logloss: 0.0237778
    [131]	valid's binary_logloss: 0.0235645
    [132]	valid's binary_logloss: 0.0233524
    [133]	valid's binary_logloss: 0.0231429
    [134]	valid's binary_logloss: 0.0229361
    [135]	valid's binary_logloss: 0.0227318
    [136]	valid's binary_logloss: 0.0225299
    [137]	valid's binary_logloss: 0.0223303
    [138]	valid's binary_logloss: 0.0221332
    [139]	valid's binary_logloss: 0.0219384
    [140]	valid's binary_logloss: 0.0217462
    [141]	valid's binary_logloss: 0.021556
    [142]	valid's binary_logloss: 0.0213687
    [143]	valid's binary_logloss: 0.0211832
    [144]	valid's binary_logloss: 0.0210004
    [145]	valid's binary_logloss: 0.0208192
    [146]	valid's binary_logloss: 0.0206403
    [147]	valid's binary_logloss: 0.0204635
    [148]	valid's binary_logloss: 0.0202887
    [149]	valid's binary_logloss: 0.0201162
    [150]	valid's binary_logloss: 0.0199456
    [151]	valid's binary_logloss: 0.019777
    [152]	valid's binary_logloss: 0.0196105
    [153]	valid's binary_logloss: 0.0194459
    [154]	valid's binary_logloss: 0.0192833
    [155]	valid's binary_logloss: 0.0191229
    [156]	valid's binary_logloss: 0.018964
    [157]	valid's binary_logloss: 0.0188071
    [158]	valid's binary_logloss: 0.0186519
    [159]	valid's binary_logloss: 0.0184988
    [160]	valid's binary_logloss: 0.0183477
    [161]	valid's binary_logloss: 0.0181981
    [162]	valid's binary_logloss: 0.0180501
    [163]	valid's binary_logloss: 0.0179037
    [164]	valid's binary_logloss: 0.017759
    [165]	valid's binary_logloss: 0.0176174
    [166]	valid's binary_logloss: 0.0174761
    [167]	valid's binary_logloss: 0.0173365
    [168]	valid's binary_logloss: 0.0171985
    [169]	valid's binary_logloss: 0.0170621
    [170]	valid's binary_logloss: 0.0169273
    [171]	valid's binary_logloss: 0.0167941
    [172]	valid's binary_logloss: 0.0166624
    [173]	valid's binary_logloss: 0.0165335
    [174]	valid's binary_logloss: 0.0164052
    [175]	valid's binary_logloss: 0.0162779
    [176]	valid's binary_logloss: 0.0161522
    [177]	valid's binary_logloss: 0.0160279
    [178]	valid's binary_logloss: 0.0159051
    [179]	valid's binary_logloss: 0.0157836
    [180]	valid's binary_logloss: 0.0156636
    [181]	valid's binary_logloss: 0.0155449
    [182]	valid's binary_logloss: 0.0154276
    [183]	valid's binary_logloss: 0.0153119
    [184]	valid's binary_logloss: 0.0151972
    [185]	valid's binary_logloss: 0.0150839
    [186]	valid's binary_logloss: 0.0149731
    [187]	valid's binary_logloss: 0.0148624
    [188]	valid's binary_logloss: 0.0147529
    [189]	valid's binary_logloss: 0.0146447
    [190]	valid's binary_logloss: 0.0145377
    [191]	valid's binary_logloss: 0.0144319
    [192]	valid's binary_logloss: 0.0143273
    [193]	valid's binary_logloss: 0.0142239
    [194]	valid's binary_logloss: 0.0141219
    [195]	valid's binary_logloss: 0.0140209
    [196]	valid's binary_logloss: 0.0139209
    [197]	valid's binary_logloss: 0.0138222
    [198]	valid's binary_logloss: 0.0137246
    [199]	valid's binary_logloss: 0.0136286
    [200]	valid's binary_logloss: 0.0135331
    Did not meet early stopping. Best iteration is:
    [200]	valid's binary_logloss: 0.0135331
    [1]	valid's binary_logloss: 0.0257666
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.0257666
    [3]	valid's binary_logloss: 0.0257666
    [4]	valid's binary_logloss: 0.0257666
    [5]	valid's binary_logloss: 0.0257666
    [6]	valid's binary_logloss: 0.0257666
    [7]	valid's binary_logloss: 0.0257666
    [8]	valid's binary_logloss: 0.0257666
    [9]	valid's binary_logloss: 0.0257666
    [10]	valid's binary_logloss: 0.0257666
    [11]	valid's binary_logloss: 0.0257666
    Early stopping, best iteration is:
    [1]	valid's binary_logloss: 0.0257666
    [1]	valid's binary_logloss: 0.0257918
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.0257918
    [3]	valid's binary_logloss: 0.0257918
    [4]	valid's binary_logloss: 0.0257918
    [5]	valid's binary_logloss: 0.0257918
    [6]	valid's binary_logloss: 0.0257918
    [7]	valid's binary_logloss: 0.0257918
    [8]	valid's binary_logloss: 0.0257918
    [9]	valid's binary_logloss: 0.0257918
    [10]	valid's binary_logloss: 0.0257918
    [11]	valid's binary_logloss: 0.0257918
    Early stopping, best iteration is:
    [1]	valid's binary_logloss: 0.0257918
    [1]	valid's binary_logloss: 0.0257889
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.0257889
    [3]	valid's binary_logloss: 0.0257889
    [4]	valid's binary_logloss: 0.0257889
    [5]	valid's binary_logloss: 0.0257889
    [6]	valid's binary_logloss: 0.0257889
    [7]	valid's binary_logloss: 0.0257889
    [8]	valid's binary_logloss: 0.0257889
    [9]	valid's binary_logloss: 0.0257889
    [10]	valid's binary_logloss: 0.0257889
    [11]	valid's binary_logloss: 0.0257889
    Early stopping, best iteration is:
    [1]	valid's binary_logloss: 0.0257889
    [1]	valid's binary_logloss: 0.031429
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.0245337
    [3]	valid's binary_logloss: 0.0195027
    [4]	valid's binary_logloss: 0.0158253
    [5]	valid's binary_logloss: 0.013146
    [6]	valid's binary_logloss: 0.0111988
    [7]	valid's binary_logloss: 0.00978399
    [8]	valid's binary_logloss: 0.00876826
    [9]	valid's binary_logloss: 0.00803863
    [10]	valid's binary_logloss: 0.00752514
    [11]	valid's binary_logloss: 0.00715963
    [12]	valid's binary_logloss: 0.00690603
    [13]	valid's binary_logloss: 0.00673966
    [14]	valid's binary_logloss: 0.00662271
    [15]	valid's binary_logloss: 0.00655317
    [16]	valid's binary_logloss: 0.00650286
    [17]	valid's binary_logloss: 0.0064783
    [18]	valid's binary_logloss: 0.00645752
    [19]	valid's binary_logloss: 0.00644502
    [20]	valid's binary_logloss: 0.00643769
    [21]	valid's binary_logloss: 0.00643517
    [22]	valid's binary_logloss: 0.00640837
    [23]	valid's binary_logloss: 0.00640531
    [24]	valid's binary_logloss: 0.00640546
    [25]	valid's binary_logloss: 0.00640609
    [26]	valid's binary_logloss: 0.00637849
    [27]	valid's binary_logloss: 0.00635992
    [28]	valid's binary_logloss: 0.00635706
    [29]	valid's binary_logloss: 0.00635564
    [30]	valid's binary_logloss: 0.00633372
    [31]	valid's binary_logloss: 0.00631877
    [32]	valid's binary_logloss: 0.00631565
    [33]	valid's binary_logloss: 0.0063164
    [34]	valid's binary_logloss: 0.00631539
    [35]	valid's binary_logloss: 0.00631499
    [36]	valid's binary_logloss: 0.00631492
    [37]	valid's binary_logloss: 0.00628786
    [38]	valid's binary_logloss: 0.00628718
    [39]	valid's binary_logloss: 0.00626567
    [40]	valid's binary_logloss: 0.0062645
    [41]	valid's binary_logloss: 0.00626569
    [42]	valid's binary_logloss: 0.00626542
    [43]	valid's binary_logloss: 0.00624272
    [44]	valid's binary_logloss: 0.00624194
    [45]	valid's binary_logloss: 0.00622337
    [46]	valid's binary_logloss: 0.00621056
    [47]	valid's binary_logloss: 0.00620368
    [48]	valid's binary_logloss: 0.00619247
    [49]	valid's binary_logloss: 0.00619251
    [50]	valid's binary_logloss: 0.00618125
    [51]	valid's binary_logloss: 0.00618233
    [52]	valid's binary_logloss: 0.00617659
    [53]	valid's binary_logloss: 0.00616471
    [54]	valid's binary_logloss: 0.00616584
    [55]	valid's binary_logloss: 0.0061611
    [56]	valid's binary_logloss: 0.0061583
    [57]	valid's binary_logloss: 0.00615371
    [58]	valid's binary_logloss: 0.00613983
    [59]	valid's binary_logloss: 0.00614096
    [60]	valid's binary_logloss: 0.00612986
    [61]	valid's binary_logloss: 0.0061196
    [62]	valid's binary_logloss: 0.00611005
    [63]	valid's binary_logloss: 0.00610895
    [64]	valid's binary_logloss: 0.00610846
    [65]	valid's binary_logloss: 0.00609709
    [66]	valid's binary_logloss: 0.00608904
    [67]	valid's binary_logloss: 0.00608254
    [68]	valid's binary_logloss: 0.00608113
    [69]	valid's binary_logloss: 0.00607186
    [70]	valid's binary_logloss: 0.00606765
    [71]	valid's binary_logloss: 0.00605975
    [72]	valid's binary_logloss: 0.00602427
    [73]	valid's binary_logloss: 0.00602367
    [74]	valid's binary_logloss: 0.00601481
    [75]	valid's binary_logloss: 0.00601425
    [76]	valid's binary_logloss: 0.00600734
    [77]	valid's binary_logloss: 0.00600199
    [78]	valid's binary_logloss: 0.00600208
    [79]	valid's binary_logloss: 0.00597098
    [80]	valid's binary_logloss: 0.00594905
    [81]	valid's binary_logloss: 0.00594081
    [82]	valid's binary_logloss: 0.00593445
    [83]	valid's binary_logloss: 0.00592878
    [84]	valid's binary_logloss: 0.00592379
    [85]	valid's binary_logloss: 0.00592564
    [86]	valid's binary_logloss: 0.00592717
    [87]	valid's binary_logloss: 0.00592023
    [88]	valid's binary_logloss: 0.0059152
    [89]	valid's binary_logloss: 0.00591549
    [90]	valid's binary_logloss: 0.00589778
    [91]	valid's binary_logloss: 0.00589766
    [92]	valid's binary_logloss: 0.00589951
    [93]	valid's binary_logloss: 0.00590104
    [94]	valid's binary_logloss: 0.00590228
    [95]	valid's binary_logloss: 0.0058958
    [96]	valid's binary_logloss: 0.00588103
    [97]	valid's binary_logloss: 0.0058829
    [98]	valid's binary_logloss: 0.005878
    [99]	valid's binary_logloss: 0.00587398
    [100]	valid's binary_logloss: 0.00587572
    [101]	valid's binary_logloss: 0.00587602
    [102]	valid's binary_logloss: 0.00587221
    [103]	valid's binary_logloss: 0.00586733
    [104]	valid's binary_logloss: 0.00586885
    [105]	valid's binary_logloss: 0.00586911
    [106]	valid's binary_logloss: 0.00587019
    [107]	valid's binary_logloss: 0.00586457
    [108]	valid's binary_logloss: 0.00586483
    [109]	valid's binary_logloss: 0.0058443
    [110]	valid's binary_logloss: 0.0058457
    [111]	valid's binary_logloss: 0.00584683
    [112]	valid's binary_logloss: 0.00584693
    [113]	valid's binary_logloss: 0.00584778
    [114]	valid's binary_logloss: 0.0058427
    [115]	valid's binary_logloss: 0.0058427
    [116]	valid's binary_logloss: 0.00584375
    [117]	valid's binary_logloss: 0.00583942
    [118]	valid's binary_logloss: 0.0058396
    [119]	valid's binary_logloss: 0.00583612
    [120]	valid's binary_logloss: 0.00583742
    [121]	valid's binary_logloss: 0.00583757
    [122]	valid's binary_logloss: 0.00583403
    [123]	valid's binary_logloss: 0.00583526
    [124]	valid's binary_logloss: 0.00583196
    [125]	valid's binary_logloss: 0.00583317
    [126]	valid's binary_logloss: 0.00583005
    [127]	valid's binary_logloss: 0.00583125
    [128]	valid's binary_logloss: 0.00582827
    [129]	valid's binary_logloss: 0.0058062
    [130]	valid's binary_logloss: 0.00580775
    [131]	valid's binary_logloss: 0.00580791
    [132]	valid's binary_logloss: 0.00580909
    [133]	valid's binary_logloss: 0.00581004
    [134]	valid's binary_logloss: 0.0058108
    [135]	valid's binary_logloss: 0.00580807
    [136]	valid's binary_logloss: 0.00580604
    [137]	valid's binary_logloss: 0.0058061
    [138]	valid's binary_logloss: 0.00580704
    [139]	valid's binary_logloss: 0.00580778
    [140]	valid's binary_logloss: 0.00580778
    [141]	valid's binary_logloss: 0.00580531
    [142]	valid's binary_logloss: 0.00580347
    [143]	valid's binary_logloss: 0.00580209
    [144]	valid's binary_logloss: 0.00580311
    [145]	valid's binary_logloss: 0.00580393
    [146]	valid's binary_logloss: 0.00580196
    [147]	valid's binary_logloss: 0.00580276
    [148]	valid's binary_logloss: 0.00580089
    [149]	valid's binary_logloss: 0.00580167
    [150]	valid's binary_logloss: 0.00580228
    [151]	valid's binary_logloss: 0.00580228
    [152]	valid's binary_logloss: 0.00580276
    [153]	valid's binary_logloss: 0.00580314
    [154]	valid's binary_logloss: 0.00580342
    [155]	valid's binary_logloss: 0.00580078
    [156]	valid's binary_logloss: 0.00580124
    [157]	valid's binary_logloss: 0.00580124
    [158]	valid's binary_logloss: 0.00580159
    [159]	valid's binary_logloss: 0.00579922
    [160]	valid's binary_logloss: 0.00579745
    [161]	valid's binary_logloss: 0.00579745
    [162]	valid's binary_logloss: 0.00579745
    [163]	valid's binary_logloss: 0.00579808
    [164]	valid's binary_logloss: 0.00579644
    [165]	valid's binary_logloss: 0.00579708
    [166]	valid's binary_logloss: 0.00579554
    [167]	valid's binary_logloss: 0.00579616
    [168]	valid's binary_logloss: 0.00579665
    [169]	valid's binary_logloss: 0.00579703
    [170]	valid's binary_logloss: 0.00579732
    [171]	valid's binary_logloss: 0.00579755
    [172]	valid's binary_logloss: 0.00579755
    [173]	valid's binary_logloss: 0.00579544
    [174]	valid's binary_logloss: 0.00579544
    [175]	valid's binary_logloss: 0.00579386
    [176]	valid's binary_logloss: 0.00579266
    [177]	valid's binary_logloss: 0.00579176
    [178]	valid's binary_logloss: 0.00579108
    [179]	valid's binary_logloss: 0.00579182
    [180]	valid's binary_logloss: 0.00579239
    [181]	valid's binary_logloss: 0.00579126
    [182]	valid's binary_logloss: 0.0057904
    [183]	valid's binary_logloss: 0.00579102
    [184]	valid's binary_logloss: 0.00579151
    [185]	valid's binary_logloss: 0.00579035
    [186]	valid's binary_logloss: 0.00578947
    [187]	valid's binary_logloss: 0.00579002
    [188]	valid's binary_logloss: 0.00579002
    [189]	valid's binary_logloss: 0.00579045
    [190]	valid's binary_logloss: 0.00579079
    [191]	valid's binary_logloss: 0.00578952
    [192]	valid's binary_logloss: 0.00578856
    [193]	valid's binary_logloss: 0.00578902
    [194]	valid's binary_logloss: 0.00578808
    [195]	valid's binary_logloss: 0.00578852
    [196]	valid's binary_logloss: 0.00578852
    [197]	valid's binary_logloss: 0.00578886
    [198]	valid's binary_logloss: 0.00578912
    [199]	valid's binary_logloss: 0.00578794
    [200]	valid's binary_logloss: 0.00578705
    Did not meet early stopping. Best iteration is:
    [200]	valid's binary_logloss: 0.00578705
    [1]	valid's binary_logloss: 0.0314244
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.024565
    [3]	valid's binary_logloss: 0.0195393
    [4]	valid's binary_logloss: 0.0158831
    [5]	valid's binary_logloss: 0.0132042
    [6]	valid's binary_logloss: 0.0112558
    [7]	valid's binary_logloss: 0.00984578
    [8]	valid's binary_logloss: 0.0088291
    [9]	valid's binary_logloss: 0.00810102
    [10]	valid's binary_logloss: 0.00758424
    [11]	valid's binary_logloss: 0.00722156
    [12]	valid's binary_logloss: 0.00697056
    [13]	valid's binary_logloss: 0.00679973
    [14]	valid's binary_logloss: 0.00668569
    [15]	valid's binary_logloss: 0.00660649
    [16]	valid's binary_logloss: 0.00655891
    [17]	valid's binary_logloss: 0.00653116
    [18]	valid's binary_logloss: 0.00651355
    [19]	valid's binary_logloss: 0.00650347
    [20]	valid's binary_logloss: 0.00649799
    [21]	valid's binary_logloss: 0.00649306
    [22]	valid's binary_logloss: 0.00646702
    [23]	valid's binary_logloss: 0.0064673
    [24]	valid's binary_logloss: 0.00646851
    [25]	valid's binary_logloss: 0.00646725
    [26]	valid's binary_logloss: 0.00644012
    [27]	valid's binary_logloss: 0.00642022
    [28]	valid's binary_logloss: 0.00642269
    [29]	valid's binary_logloss: 0.0064234
    [30]	valid's binary_logloss: 0.00640125
    [31]	valid's binary_logloss: 0.00638493
    [32]	valid's binary_logloss: 0.00638777
    [33]	valid's binary_logloss: 0.00639089
    [34]	valid's binary_logloss: 0.00639207
    [35]	valid's binary_logloss: 0.00639336
    [36]	valid's binary_logloss: 0.00639537
    [37]	valid's binary_logloss: 0.00636818
    [38]	valid's binary_logloss: 0.00636942
    [39]	valid's binary_logloss: 0.00634642
    [40]	valid's binary_logloss: 0.00634762
    [41]	valid's binary_logloss: 0.00634792
    [42]	valid's binary_logloss: 0.00634924
    [43]	valid's binary_logloss: 0.00632424
    [44]	valid's binary_logloss: 0.00632553
    [45]	valid's binary_logloss: 0.00630424
    [46]	valid's binary_logloss: 0.00628858
    [47]	valid's binary_logloss: 0.00628962
    [48]	valid's binary_logloss: 0.00627428
    [49]	valid's binary_logloss: 0.0062754
    [50]	valid's binary_logloss: 0.00626043
    [51]	valid's binary_logloss: 0.00626036
    [52]	valid's binary_logloss: 0.00624654
    [53]	valid's binary_logloss: 0.00624943
    [54]	valid's binary_logloss: 0.00623596
    [55]	valid's binary_logloss: 0.0062371
    [56]	valid's binary_logloss: 0.00623849
    [57]	valid's binary_logloss: 0.00623992
    [58]	valid's binary_logloss: 0.006221
    [59]	valid's binary_logloss: 0.00622371
    [60]	valid's binary_logloss: 0.00620765
    [61]	valid's binary_logloss: 0.006209
    [62]	valid's binary_logloss: 0.00619425
    [63]	valid's binary_logloss: 0.00619698
    [64]	valid's binary_logloss: 0.0061994
    [65]	valid's binary_logloss: 0.00618344
    [66]	valid's binary_logloss: 0.00617158
    [67]	valid's binary_logloss: 0.00617363
    [68]	valid's binary_logloss: 0.00617564
    [69]	valid's binary_logloss: 0.00616089
    [70]	valid's binary_logloss: 0.00616292
    [71]	valid's binary_logloss: 0.00616428
    [72]	valid's binary_logloss: 0.00614837
    [73]	valid's binary_logloss: 0.00615059
    [74]	valid's binary_logloss: 0.00613681
    [75]	valid's binary_logloss: 0.006139
    [76]	valid's binary_logloss: 0.00614036
    [77]	valid's binary_logloss: 0.00614169
    [78]	valid's binary_logloss: 0.00614311
    [79]	valid's binary_logloss: 0.0061257
    [80]	valid's binary_logloss: 0.00611281
    [81]	valid's binary_logloss: 0.00611417
    [82]	valid's binary_logloss: 0.00611553
    [83]	valid's binary_logloss: 0.00610167
    [84]	valid's binary_logloss: 0.00609132
    [85]	valid's binary_logloss: 0.00609353
    [86]	valid's binary_logloss: 0.00609541
    [87]	valid's binary_logloss: 0.00608419
    [88]	valid's binary_logloss: 0.00607576
    [89]	valid's binary_logloss: 0.00607713
    [90]	valid's binary_logloss: 0.00607856
    [91]	valid's binary_logloss: 0.00608026
    [92]	valid's binary_logloss: 0.00606826
    [93]	valid's binary_logloss: 0.00607012
    [94]	valid's binary_logloss: 0.00607148
    [95]	valid's binary_logloss: 0.00605966
    [96]	valid's binary_logloss: 0.0060508
    [97]	valid's binary_logloss: 0.00605222
    [98]	valid's binary_logloss: 0.0060436
    [99]	valid's binary_logloss: 0.00603707
    [100]	valid's binary_logloss: 0.00603933
    [101]	valid's binary_logloss: 0.00604126
    [102]	valid's binary_logloss: 0.006043
    [103]	valid's binary_logloss: 0.00603314
    [104]	valid's binary_logloss: 0.0060349
    [105]	valid's binary_logloss: 0.00603645
    [106]	valid's binary_logloss: 0.00603715
    [107]	valid's binary_logloss: 0.00602636
    [108]	valid's binary_logloss: 0.00602782
    [109]	valid's binary_logloss: 0.00601863
    [110]	valid's binary_logloss: 0.00601167
    [111]	valid's binary_logloss: 0.00601303
    [112]	valid's binary_logloss: 0.00601457
    [113]	valid's binary_logloss: 0.00601578
    [114]	valid's binary_logloss: 0.00600664
    [115]	valid's binary_logloss: 0.00600769
    [116]	valid's binary_logloss: 0.00600888
    [117]	valid's binary_logloss: 0.0060003
    [118]	valid's binary_logloss: 0.00600184
    [119]	valid's binary_logloss: 0.00599407
    [120]	valid's binary_logloss: 0.00599508
    [121]	valid's binary_logloss: 0.00599656
    [122]	valid's binary_logloss: 0.00598896
    [123]	valid's binary_logloss: 0.00599046
    [124]	valid's binary_logloss: 0.00598348
    [125]	valid's binary_logloss: 0.00598157
    [126]	valid's binary_logloss: 0.00597598
    [127]	valid's binary_logloss: 0.00597364
    [128]	valid's binary_logloss: 0.00597207
    [129]	valid's binary_logloss: 0.005967
    [130]	valid's binary_logloss: 0.00596511
    [131]	valid's binary_logloss: 0.00596519
    [132]	valid's binary_logloss: 0.00596526
    [133]	valid's binary_logloss: 0.00596396
    [134]	valid's binary_logloss: 0.00596241
    [135]	valid's binary_logloss: 0.00594285
    [136]	valid's binary_logloss: 0.00594133
    [137]	valid's binary_logloss: 0.0059402
    [138]	valid's binary_logloss: 0.00593936
    [139]	valid's binary_logloss: 0.00594016
    [140]	valid's binary_logloss: 0.00593942
    [141]	valid's binary_logloss: 0.00592281
    [142]	valid's binary_logloss: 0.00592006
    [143]	valid's binary_logloss: 0.0059195
    [144]	valid's binary_logloss: 0.00590731
    [145]	valid's binary_logloss: 0.00590671
    [146]	valid's binary_logloss: 0.00590627
    [147]	valid's binary_logloss: 0.00590595
    [148]	valid's binary_logloss: 0.0059057
    [149]	valid's binary_logloss: 0.00589582
    [150]	valid's binary_logloss: 0.00589548
    [151]	valid's binary_logloss: 0.00589523
    [152]	valid's binary_logloss: 0.00589563
    [153]	valid's binary_logloss: 0.00589541
    [154]	valid's binary_logloss: 0.00589524
    [155]	valid's binary_logloss: 0.00589563
    [156]	valid's binary_logloss: 0.00589594
    [157]	valid's binary_logloss: 0.00589619
    [158]	valid's binary_logloss: 0.00589639
    [159]	valid's binary_logloss: 0.00589546
    [160]	valid's binary_logloss: 0.00589473
    [161]	valid's binary_logloss: 0.005895
    [162]	valid's binary_logloss: 0.00589484
    [163]	valid's binary_logloss: 0.00589508
    [164]	valid's binary_logloss: 0.00588711
    [165]	valid's binary_logloss: 0.00588718
    [166]	valid's binary_logloss: 0.00588679
    [167]	valid's binary_logloss: 0.00588652
    [168]	valid's binary_logloss: 0.00588657
    [169]	valid's binary_logloss: 0.00588637
    [170]	valid's binary_logloss: 0.00588642
    [171]	valid's binary_logloss: 0.00588626
    [172]	valid's binary_logloss: 0.00588613
    [173]	valid's binary_logloss: 0.00588623
    [174]	valid's binary_logloss: 0.00588631
    [175]	valid's binary_logloss: 0.00588619
    [176]	valid's binary_logloss: 0.00588627
    [177]	valid's binary_logloss: 0.00588627
    [178]	valid's binary_logloss: 0.00588616
    [179]	valid's binary_logloss: 0.00588607
    [180]	valid's binary_logloss: 0.00588618
    [181]	valid's binary_logloss: 0.00588617
    [182]	valid's binary_logloss: 0.00588608
    [183]	valid's binary_logloss: 0.00588619
    [184]	valid's binary_logloss: 0.00588627
    [185]	valid's binary_logloss: 0.00588627
    [186]	valid's binary_logloss: 0.00588633
    [187]	valid's binary_logloss: 0.00588633
    [188]	valid's binary_logloss: 0.00588633
    [189]	valid's binary_logloss: 0.00588633
    Early stopping, best iteration is:
    [179]	valid's binary_logloss: 0.00588607
    [1]	valid's binary_logloss: 0.0314382
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.024545
    [3]	valid's binary_logloss: 0.0195126
    [4]	valid's binary_logloss: 0.0158407
    [5]	valid's binary_logloss: 0.0131669
    [6]	valid's binary_logloss: 0.0112203
    [7]	valid's binary_logloss: 0.00981168
    [8]	valid's binary_logloss: 0.0087948
    [9]	valid's binary_logloss: 0.00806441
    [10]	valid's binary_logloss: 0.00754115
    [11]	valid's binary_logloss: 0.00717463
    [12]	valid's binary_logloss: 0.00692008
    [13]	valid's binary_logloss: 0.00674513
    [14]	valid's binary_logloss: 0.00662723
    [15]	valid's binary_logloss: 0.00655012
    [16]	valid's binary_logloss: 0.006499
    [17]	valid's binary_logloss: 0.00646831
    [18]	valid's binary_logloss: 0.00644769
    [19]	valid's binary_logloss: 0.00643502
    [20]	valid's binary_logloss: 0.00642737
    [21]	valid's binary_logloss: 0.00642401
    [22]	valid's binary_logloss: 0.00639001
    [23]	valid's binary_logloss: 0.00638733
    [24]	valid's binary_logloss: 0.00638727
    [25]	valid's binary_logloss: 0.00638774
    [26]	valid's binary_logloss: 0.00635623
    [27]	valid's binary_logloss: 0.00633491
    [28]	valid's binary_logloss: 0.00633318
    [29]	valid's binary_logloss: 0.00633267
    [30]	valid's binary_logloss: 0.00630907
    [31]	valid's binary_logloss: 0.00629285
    [32]	valid's binary_logloss: 0.0062911
    [33]	valid's binary_logloss: 0.00629227
    [34]	valid's binary_logloss: 0.0062922
    [35]	valid's binary_logloss: 0.00629263
    [36]	valid's binary_logloss: 0.00629316
    [37]	valid's binary_logloss: 0.00626645
    [38]	valid's binary_logloss: 0.00626267
    [39]	valid's binary_logloss: 0.00624288
    [40]	valid's binary_logloss: 0.00624268
    [41]	valid's binary_logloss: 0.00624294
    [42]	valid's binary_logloss: 0.0062404
    [43]	valid's binary_logloss: 0.00621943
    [44]	valid's binary_logloss: 0.0062196
    [45]	valid's binary_logloss: 0.00620239
    [46]	valid's binary_logloss: 0.00619036
    [47]	valid's binary_logloss: 0.00618958
    [48]	valid's binary_logloss: 0.00617792
    [49]	valid's binary_logloss: 0.0061774
    [50]	valid's binary_logloss: 0.00616626
    [51]	valid's binary_logloss: 0.00616594
    [52]	valid's binary_logloss: 0.0061554
    [53]	valid's binary_logloss: 0.00615478
    [54]	valid's binary_logloss: 0.00614436
    [55]	valid's binary_logloss: 0.00614389
    [56]	valid's binary_logloss: 0.00614411
    [57]	valid's binary_logloss: 0.00613985
    [58]	valid's binary_logloss: 0.00612669
    [59]	valid's binary_logloss: 0.00612695
    [60]	valid's binary_logloss: 0.00611571
    [61]	valid's binary_logloss: 0.00610786
    [62]	valid's binary_logloss: 0.00609863
    [63]	valid's binary_logloss: 0.00609868
    [64]	valid's binary_logloss: 0.00609898
    [65]	valid's binary_logloss: 0.00608851
    [66]	valid's binary_logloss: 0.00608099
    [67]	valid's binary_logloss: 0.00607633
    [68]	valid's binary_logloss: 0.00606917
    [69]	valid's binary_logloss: 0.00606186
    [70]	valid's binary_logloss: 0.00606194
    [71]	valid's binary_logloss: 0.00605771
    [72]	valid's binary_logloss: 0.00602617
    [73]	valid's binary_logloss: 0.00602618
    [74]	valid's binary_logloss: 0.00601965
    [75]	valid's binary_logloss: 0.00601971
    [76]	valid's binary_logloss: 0.00601367
    [77]	valid's binary_logloss: 0.00600908
    [78]	valid's binary_logloss: 0.0060095
    [79]	valid's binary_logloss: 0.00598153
    [80]	valid's binary_logloss: 0.00596129
    [81]	valid's binary_logloss: 0.00592578
    [82]	valid's binary_logloss: 0.00592139
    [83]	valid's binary_logloss: 0.0059061
    [84]	valid's binary_logloss: 0.00590651
    [85]	valid's binary_logloss: 0.00590675
    [86]	valid's binary_logloss: 0.00590709
    [87]	valid's binary_logloss: 0.0058946
    [88]	valid's binary_logloss: 0.00589468
    [89]	valid's binary_logloss: 0.00589478
    [90]	valid's binary_logloss: 0.00587488
    [91]	valid's binary_logloss: 0.00587506
    [92]	valid's binary_logloss: 0.0058752
    [93]	valid's binary_logloss: 0.00587532
    [94]	valid's binary_logloss: 0.00587549
    [95]	valid's binary_logloss: 0.00587462
    [96]	valid's binary_logloss: 0.00587477
    [97]	valid's binary_logloss: 0.00587491
    [98]	valid's binary_logloss: 0.00587376
    [99]	valid's binary_logloss: 0.00587388
    [100]	valid's binary_logloss: 0.00587281
    [101]	valid's binary_logloss: 0.00587293
    [102]	valid's binary_logloss: 0.00587303
    [103]	valid's binary_logloss: 0.00587188
    [104]	valid's binary_logloss: 0.005871
    [105]	valid's binary_logloss: 0.00587111
    [106]	valid's binary_logloss: 0.00587121
    [107]	valid's binary_logloss: 0.00587023
    [108]	valid's binary_logloss: 0.00587032
    [109]	valid's binary_logloss: 0.00586944
    [110]	valid's binary_logloss: 0.00586877
    [111]	valid's binary_logloss: 0.00586878
    [112]	valid's binary_logloss: 0.00586888
    [113]	valid's binary_logloss: 0.00586896
    [114]	valid's binary_logloss: 0.00586817
    [115]	valid's binary_logloss: 0.00586817
    [116]	valid's binary_logloss: 0.00586825
    [117]	valid's binary_logloss: 0.00586753
    [118]	valid's binary_logloss: 0.00586761
    [119]	valid's binary_logloss: 0.00586695
    [120]	valid's binary_logloss: 0.00586703
    [121]	valid's binary_logloss: 0.00586708
    [122]	valid's binary_logloss: 0.0058664
    [123]	valid's binary_logloss: 0.0058664
    [124]	valid's binary_logloss: 0.00586587
    [125]	valid's binary_logloss: 0.00586547
    [126]	valid's binary_logloss: 0.00586516
    [127]	valid's binary_logloss: 0.00586521
    [128]	valid's binary_logloss: 0.00586529
    [129]	valid's binary_logloss: 0.00586536
    [130]	valid's binary_logloss: 0.00586541
    [131]	valid's binary_logloss: 0.00586545
    [132]	valid's binary_logloss: 0.00586545
    [133]	valid's binary_logloss: 0.00586485
    [134]	valid's binary_logloss: 0.0058649
    [135]	valid's binary_logloss: 0.00586438
    [136]	valid's binary_logloss: 0.00586398
    [137]	valid's binary_logloss: 0.00586404
    [138]	valid's binary_logloss: 0.00586404
    [139]	valid's binary_logloss: 0.00586408
    [140]	valid's binary_logloss: 0.00586408
    [141]	valid's binary_logloss: 0.00586365
    [142]	valid's binary_logloss: 0.00586331
    [143]	valid's binary_logloss: 0.00586306
    [144]	valid's binary_logloss: 0.00586286
    [145]	valid's binary_logloss: 0.00586271
    [146]	valid's binary_logloss: 0.00586277
    [147]	valid's binary_logloss: 0.00586282
    [148]	valid's binary_logloss: 0.00586255
    [149]	valid's binary_logloss: 0.0058626
    [150]	valid's binary_logloss: 0.00586263
    [151]	valid's binary_logloss: 0.00586266
    [152]	valid's binary_logloss: 0.00586266
    [153]	valid's binary_logloss: 0.00586266
    [154]	valid's binary_logloss: 0.00586268
    [155]	valid's binary_logloss: 0.00586231
    [156]	valid's binary_logloss: 0.00586234
    [157]	valid's binary_logloss: 0.00586234
    [158]	valid's binary_logloss: 0.00586234
    [159]	valid's binary_logloss: 0.00586202
    [160]	valid's binary_logloss: 0.00586177
    [161]	valid's binary_logloss: 0.0058618
    [162]	valid's binary_logloss: 0.00586183
    [163]	valid's binary_logloss: 0.00586185
    [164]	valid's binary_logloss: 0.00586156
    [165]	valid's binary_logloss: 0.00586156
    [166]	valid's binary_logloss: 0.00586134
    [167]	valid's binary_logloss: 0.00586137
    [168]	valid's binary_logloss: 0.00586137
    [169]	valid's binary_logloss: 0.00586139
    [170]	valid's binary_logloss: 0.00586141
    [171]	valid's binary_logloss: 0.00586142
    [172]	valid's binary_logloss: 0.00586142
    [173]	valid's binary_logloss: 0.00586115
    [174]	valid's binary_logloss: 0.00586117
    [175]	valid's binary_logloss: 0.00586094
    [176]	valid's binary_logloss: 0.00586076
    [177]	valid's binary_logloss: 0.00586063
    [178]	valid's binary_logloss: 0.00586066
    [179]	valid's binary_logloss: 0.00586052
    [180]	valid's binary_logloss: 0.00586054
    [181]	valid's binary_logloss: 0.00586041
    [182]	valid's binary_logloss: 0.00586043
    [183]	valid's binary_logloss: 0.00586045
    [184]	valid's binary_logloss: 0.00586046
    [185]	valid's binary_logloss: 0.0058603
    [186]	valid's binary_logloss: 0.00586017
    [187]	valid's binary_logloss: 0.00586019
    [188]	valid's binary_logloss: 0.00586021
    [189]	valid's binary_logloss: 0.00586008
    [190]	valid's binary_logloss: 0.0058601
    [191]	valid's binary_logloss: 0.00585999
    [192]	valid's binary_logloss: 0.0058599
    [193]	valid's binary_logloss: 0.00585994
    [194]	valid's binary_logloss: 0.00585985
    [195]	valid's binary_logloss: 0.00585987
    [196]	valid's binary_logloss: 0.00585987
    [197]	valid's binary_logloss: 0.00585988
    [198]	valid's binary_logloss: 0.00585989
    [199]	valid's binary_logloss: 0.00585978
    [200]	valid's binary_logloss: 0.00585969
    Did not meet early stopping. Best iteration is:
    [200]	valid's binary_logloss: 0.00585969
    [1]	valid's binary_logloss: 0.0753546
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.0656279
    [3]	valid's binary_logloss: 0.0579287
    [4]	valid's binary_logloss: 0.0515816
    [5]	valid's binary_logloss: 0.0462736
    [6]	valid's binary_logloss: 0.0417216
    [7]	valid's binary_logloss: 0.0377542
    [8]	valid's binary_logloss: 0.0343012
    [9]	valid's binary_logloss: 0.0312746
    [10]	valid's binary_logloss: 0.0285961
    [11]	valid's binary_logloss: 0.0262291
    [12]	valid's binary_logloss: 0.0241391
    [13]	valid's binary_logloss: 0.0222741
    [14]	valid's binary_logloss: 0.0206235
    [15]	valid's binary_logloss: 0.0191667
    [16]	valid's binary_logloss: 0.0178474
    [17]	valid's binary_logloss: 0.0166859
    [18]	valid's binary_logloss: 0.0156307
    [19]	valid's binary_logloss: 0.0146821
    [20]	valid's binary_logloss: 0.0138337
    [21]	valid's binary_logloss: 0.0130747
    [22]	valid's binary_logloss: 0.0124157
    [23]	valid's binary_logloss: 0.0118051
    [24]	valid's binary_logloss: 0.0112661
    [25]	valid's binary_logloss: 0.0107812
    [26]	valid's binary_logloss: 0.0103387
    [27]	valid's binary_logloss: 0.009944
    [28]	valid's binary_logloss: 0.00959111
    [29]	valid's binary_logloss: 0.00927186
    [30]	valid's binary_logloss: 0.00900322
    [31]	valid's binary_logloss: 0.00876373
    [32]	valid's binary_logloss: 0.0085338
    [33]	valid's binary_logloss: 0.00833294
    [34]	valid's binary_logloss: 0.00814466
    [35]	valid's binary_logloss: 0.0079759
    [36]	valid's binary_logloss: 0.00782485
    [37]	valid's binary_logloss: 0.00768907
    [38]	valid's binary_logloss: 0.00756735
    [39]	valid's binary_logloss: 0.00746528
    [40]	valid's binary_logloss: 0.00736667
    [41]	valid's binary_logloss: 0.00728655
    [42]	valid's binary_logloss: 0.0072066
    [43]	valid's binary_logloss: 0.00713332
    [44]	valid's binary_logloss: 0.00706893
    [45]	valid's binary_logloss: 0.00700953
    [46]	valid's binary_logloss: 0.00696157
    [47]	valid's binary_logloss: 0.00691314
    [48]	valid's binary_logloss: 0.00687061
    [49]	valid's binary_logloss: 0.00683125
    [50]	valid's binary_logloss: 0.00679684
    [51]	valid's binary_logloss: 0.00676811
    [52]	valid's binary_logloss: 0.00673904
    [53]	valid's binary_logloss: 0.0067136
    [54]	valid's binary_logloss: 0.00669449
    [55]	valid's binary_logloss: 0.00667342
    [56]	valid's binary_logloss: 0.00665374
    [57]	valid's binary_logloss: 0.0066366
    [58]	valid's binary_logloss: 0.00662052
    [59]	valid's binary_logloss: 0.00660966
    [60]	valid's binary_logloss: 0.0065967
    [61]	valid's binary_logloss: 0.00658456
    [62]	valid's binary_logloss: 0.00657398
    [63]	valid's binary_logloss: 0.00656445
    [64]	valid's binary_logloss: 0.00655587
    [65]	valid's binary_logloss: 0.00654813
    [66]	valid's binary_logloss: 0.00654115
    [67]	valid's binary_logloss: 0.00653437
    [68]	valid's binary_logloss: 0.00652824
    [69]	valid's binary_logloss: 0.00652271
    [70]	valid's binary_logloss: 0.00651772
    [71]	valid's binary_logloss: 0.00651321
    [72]	valid's binary_logloss: 0.00650914
    [73]	valid's binary_logloss: 0.00650566
    [74]	valid's binary_logloss: 0.00650252
    [75]	valid's binary_logloss: 0.00649969
    [76]	valid's binary_logloss: 0.00649691
    [77]	valid's binary_logloss: 0.0064944
    [78]	valid's binary_logloss: 0.0064923
    [79]	valid's binary_logloss: 0.00649023
    [80]	valid's binary_logloss: 0.00648836
    [81]	valid's binary_logloss: 0.00648667
    [82]	valid's binary_logloss: 0.00648514
    [83]	valid's binary_logloss: 0.00648387
    [84]	valid's binary_logloss: 0.00648261
    [85]	valid's binary_logloss: 0.00648157
    [86]	valid's binary_logloss: 0.00648063
    [87]	valid's binary_logloss: 0.00647978
    [88]	valid's binary_logloss: 0.00647901
    [89]	valid's binary_logloss: 0.00647822
    [90]	valid's binary_logloss: 0.00647749
    [91]	valid's binary_logloss: 0.00647693
    [92]	valid's binary_logloss: 0.00647633
    [93]	valid's binary_logloss: 0.00647633
    [94]	valid's binary_logloss: 0.00647579
    [95]	valid's binary_logloss: 0.00647554
    [96]	valid's binary_logloss: 0.0064751
    [97]	valid's binary_logloss: 0.0064747
    [98]	valid's binary_logloss: 0.00647432
    [99]	valid's binary_logloss: 0.0064742
    [100]	valid's binary_logloss: 0.00647409
    [101]	valid's binary_logloss: 0.00647379
    [102]	valid's binary_logloss: 0.00647349
    [103]	valid's binary_logloss: 0.00647345
    [104]	valid's binary_logloss: 0.00647318
    [105]	valid's binary_logloss: 0.00647296
    [106]	valid's binary_logloss: 0.00647296
    [107]	valid's binary_logloss: 0.00647278
    [108]	valid's binary_logloss: 0.00647261
    [109]	valid's binary_logloss: 0.00647247
    [110]	valid's binary_logloss: 0.00647233
    [111]	valid's binary_logloss: 0.00647218
    [112]	valid's binary_logloss: 0.00647207
    [113]	valid's binary_logloss: 0.00647194
    [114]	valid's binary_logloss: 0.00647182
    [115]	valid's binary_logloss: 0.00647182
    [116]	valid's binary_logloss: 0.00647171
    [117]	valid's binary_logloss: 0.00647171
    [118]	valid's binary_logloss: 0.00647161
    [119]	valid's binary_logloss: 0.00647161
    [120]	valid's binary_logloss: 0.00647161
    [121]	valid's binary_logloss: 0.00647153
    [122]	valid's binary_logloss: 0.00647145
    [123]	valid's binary_logloss: 0.00647137
    [124]	valid's binary_logloss: 0.00647131
    [125]	valid's binary_logloss: 0.00647125
    [126]	valid's binary_logloss: 0.0064712
    [127]	valid's binary_logloss: 0.00647115
    [128]	valid's binary_logloss: 0.00647111
    [129]	valid's binary_logloss: 0.00647107
    [130]	valid's binary_logloss: 0.00647103
    [131]	valid's binary_logloss: 0.00647103
    [132]	valid's binary_logloss: 0.00647103
    [133]	valid's binary_logloss: 0.006471
    [134]	valid's binary_logloss: 0.00647097
    [135]	valid's binary_logloss: 0.00647094
    [136]	valid's binary_logloss: 0.00647092
    [137]	valid's binary_logloss: 0.0064709
    [138]	valid's binary_logloss: 0.00647088
    [139]	valid's binary_logloss: 0.00647087
    [140]	valid's binary_logloss: 0.00647086
    [141]	valid's binary_logloss: 0.00647084
    [142]	valid's binary_logloss: 0.00647084
    [143]	valid's binary_logloss: 0.00647083
    [144]	valid's binary_logloss: 0.00647082
    [145]	valid's binary_logloss: 0.00647081
    [146]	valid's binary_logloss: 0.0064708
    [147]	valid's binary_logloss: 0.00647079
    [148]	valid's binary_logloss: 0.00647078
    [149]	valid's binary_logloss: 0.00647078
    [150]	valid's binary_logloss: 0.00647077
    [151]	valid's binary_logloss: 0.00647077
    [152]	valid's binary_logloss: 0.00647077
    [153]	valid's binary_logloss: 0.00647076
    [154]	valid's binary_logloss: 0.00647075
    [155]	valid's binary_logloss: 0.00647075
    [156]	valid's binary_logloss: 0.00647075
    [157]	valid's binary_logloss: 0.00647074
    [158]	valid's binary_logloss: 0.00647074
    [159]	valid's binary_logloss: 0.00647074
    [160]	valid's binary_logloss: 0.00647074
    [161]	valid's binary_logloss: 0.00647074
    [162]	valid's binary_logloss: 0.00647073
    [163]	valid's binary_logloss: 0.00647073
    [164]	valid's binary_logloss: 0.00647073
    [165]	valid's binary_logloss: 0.00647073
    [166]	valid's binary_logloss: 0.00647073
    [167]	valid's binary_logloss: 0.00647072
    [168]	valid's binary_logloss: 0.00647072
    [169]	valid's binary_logloss: 0.00647072
    [170]	valid's binary_logloss: 0.00647072
    [171]	valid's binary_logloss: 0.00647072
    [172]	valid's binary_logloss: 0.00647072
    [173]	valid's binary_logloss: 0.00647072
    [174]	valid's binary_logloss: 0.00647072
    [175]	valid's binary_logloss: 0.00647072
    [176]	valid's binary_logloss: 0.00647071
    [177]	valid's binary_logloss: 0.00647071
    [178]	valid's binary_logloss: 0.00647071
    [179]	valid's binary_logloss: 0.00647071
    [180]	valid's binary_logloss: 0.00647071
    [181]	valid's binary_logloss: 0.00647071
    [182]	valid's binary_logloss: 0.00647071
    [183]	valid's binary_logloss: 0.00647071
    [184]	valid's binary_logloss: 0.00647071
    [185]	valid's binary_logloss: 0.00647071
    [186]	valid's binary_logloss: 0.00647071
    [187]	valid's binary_logloss: 0.00647071
    [188]	valid's binary_logloss: 0.00647071
    [189]	valid's binary_logloss: 0.00647071
    [190]	valid's binary_logloss: 0.0064707
    [191]	valid's binary_logloss: 0.0064707
    [192]	valid's binary_logloss: 0.0064707
    [193]	valid's binary_logloss: 0.0064707
    [194]	valid's binary_logloss: 0.0064707
    [195]	valid's binary_logloss: 0.0064707
    [196]	valid's binary_logloss: 0.0064707
    [197]	valid's binary_logloss: 0.0064707
    [198]	valid's binary_logloss: 0.0064707
    [199]	valid's binary_logloss: 0.0064707
    [200]	valid's binary_logloss: 0.0064707
    [201]	valid's binary_logloss: 0.0064707
    [202]	valid's binary_logloss: 0.0064707
    [203]	valid's binary_logloss: 0.0064707
    [204]	valid's binary_logloss: 0.0064707
    [205]	valid's binary_logloss: 0.0064707
    [206]	valid's binary_logloss: 0.0064707
    [207]	valid's binary_logloss: 0.0064707
    [208]	valid's binary_logloss: 0.0064707
    [209]	valid's binary_logloss: 0.0064707
    [210]	valid's binary_logloss: 0.0064707
    [211]	valid's binary_logloss: 0.0064707
    [212]	valid's binary_logloss: 0.0064707
    [213]	valid's binary_logloss: 0.0064707
    [214]	valid's binary_logloss: 0.0064707
    [215]	valid's binary_logloss: 0.0064707
    [216]	valid's binary_logloss: 0.0064707
    [217]	valid's binary_logloss: 0.0064707
    [218]	valid's binary_logloss: 0.0064707
    [219]	valid's binary_logloss: 0.0064707
    [220]	valid's binary_logloss: 0.0064707
    [221]	valid's binary_logloss: 0.0064707
    [222]	valid's binary_logloss: 0.0064707
    [223]	valid's binary_logloss: 0.0064707
    [224]	valid's binary_logloss: 0.0064707
    [225]	valid's binary_logloss: 0.0064707
    [226]	valid's binary_logloss: 0.0064707
    [227]	valid's binary_logloss: 0.0064707
    [228]	valid's binary_logloss: 0.0064707
    [229]	valid's binary_logloss: 0.0064707
    [230]	valid's binary_logloss: 0.0064707
    [231]	valid's binary_logloss: 0.0064707
    [232]	valid's binary_logloss: 0.0064707
    [233]	valid's binary_logloss: 0.0064707
    [234]	valid's binary_logloss: 0.0064707
    [235]	valid's binary_logloss: 0.0064707
    [236]	valid's binary_logloss: 0.0064707
    [237]	valid's binary_logloss: 0.0064707
    [238]	valid's binary_logloss: 0.0064707
    [239]	valid's binary_logloss: 0.0064707
    [240]	valid's binary_logloss: 0.0064707
    [241]	valid's binary_logloss: 0.0064707
    [242]	valid's binary_logloss: 0.0064707
    [243]	valid's binary_logloss: 0.0064707
    [244]	valid's binary_logloss: 0.0064707
    [245]	valid's binary_logloss: 0.0064707
    [246]	valid's binary_logloss: 0.0064707
    [247]	valid's binary_logloss: 0.0064707
    [248]	valid's binary_logloss: 0.0064707
    [249]	valid's binary_logloss: 0.0064707
    [250]	valid's binary_logloss: 0.0064707
    [251]	valid's binary_logloss: 0.0064707
    [252]	valid's binary_logloss: 0.0064707
    [253]	valid's binary_logloss: 0.0064707
    [254]	valid's binary_logloss: 0.0064707
    [255]	valid's binary_logloss: 0.0064707
    [256]	valid's binary_logloss: 0.0064707
    [257]	valid's binary_logloss: 0.0064707
    [258]	valid's binary_logloss: 0.0064707
    [259]	valid's binary_logloss: 0.0064707
    [260]	valid's binary_logloss: 0.0064707
    [261]	valid's binary_logloss: 0.0064707
    [262]	valid's binary_logloss: 0.0064707
    [263]	valid's binary_logloss: 0.0064707
    [264]	valid's binary_logloss: 0.0064707
    [265]	valid's binary_logloss: 0.0064707
    [266]	valid's binary_logloss: 0.0064707
    [267]	valid's binary_logloss: 0.0064707
    [268]	valid's binary_logloss: 0.0064707
    [269]	valid's binary_logloss: 0.0064707
    [270]	valid's binary_logloss: 0.0064707
    [271]	valid's binary_logloss: 0.0064707
    [272]	valid's binary_logloss: 0.0064707
    [273]	valid's binary_logloss: 0.0064707
    [274]	valid's binary_logloss: 0.0064707
    [275]	valid's binary_logloss: 0.0064707
    [276]	valid's binary_logloss: 0.0064707
    [277]	valid's binary_logloss: 0.0064707
    [278]	valid's binary_logloss: 0.0064707
    [279]	valid's binary_logloss: 0.0064707
    [280]	valid's binary_logloss: 0.0064707
    [281]	valid's binary_logloss: 0.0064707
    [282]	valid's binary_logloss: 0.0064707
    [283]	valid's binary_logloss: 0.0064707
    [284]	valid's binary_logloss: 0.0064707
    [285]	valid's binary_logloss: 0.0064707
    [286]	valid's binary_logloss: 0.0064707
    [287]	valid's binary_logloss: 0.0064707
    [288]	valid's binary_logloss: 0.0064707
    [289]	valid's binary_logloss: 0.0064707
    [290]	valid's binary_logloss: 0.0064707
    [291]	valid's binary_logloss: 0.0064707
    [292]	valid's binary_logloss: 0.0064707
    [293]	valid's binary_logloss: 0.0064707
    Early stopping, best iteration is:
    [283]	valid's binary_logloss: 0.0064707
    [1]	valid's binary_logloss: 0.07518
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.0654762
    [3]	valid's binary_logloss: 0.0577905
    [4]	valid's binary_logloss: 0.0514655
    [5]	valid's binary_logloss: 0.0461616
    [6]	valid's binary_logloss: 0.0416221
    [7]	valid's binary_logloss: 0.0376617
    [8]	valid's binary_logloss: 0.034214
    [9]	valid's binary_logloss: 0.0311904
    [10]	valid's binary_logloss: 0.0285155
    [11]	valid's binary_logloss: 0.0261517
    [12]	valid's binary_logloss: 0.0240637
    [13]	valid's binary_logloss: 0.0222016
    [14]	valid's binary_logloss: 0.0205523
    [15]	valid's binary_logloss: 0.0190971
    [16]	valid's binary_logloss: 0.0177835
    [17]	valid's binary_logloss: 0.0166318
    [18]	valid's binary_logloss: 0.0155787
    [19]	valid's binary_logloss: 0.014632
    [20]	valid's binary_logloss: 0.0137854
    [21]	valid's binary_logloss: 0.013028
    [22]	valid's binary_logloss: 0.0123695
    [23]	valid's binary_logloss: 0.0117582
    [24]	valid's binary_logloss: 0.0112335
    [25]	valid's binary_logloss: 0.0107614
    [26]	valid's binary_logloss: 0.0103201
    [27]	valid's binary_logloss: 0.00992654
    [28]	valid's binary_logloss: 0.00957554
    [29]	valid's binary_logloss: 0.00925385
    [30]	valid's binary_logloss: 0.00898145
    [31]	valid's binary_logloss: 0.00873934
    [32]	valid's binary_logloss: 0.00851035
    [33]	valid's binary_logloss: 0.00831626
    [34]	valid's binary_logloss: 0.00812567
    [35]	valid's binary_logloss: 0.00795485
    [36]	valid's binary_logloss: 0.00780653
    [37]	valid's binary_logloss: 0.00766873
    [38]	valid's binary_logloss: 0.0075452
    [39]	valid's binary_logloss: 0.00744485
    [40]	valid's binary_logloss: 0.00734447
    [41]	valid's binary_logloss: 0.00726465
    [42]	valid's binary_logloss: 0.00718549
    [43]	valid's binary_logloss: 0.00711182
    [44]	valid's binary_logloss: 0.0070457
    [45]	valid's binary_logloss: 0.00698855
    [46]	valid's binary_logloss: 0.0069459
    [47]	valid's binary_logloss: 0.00689705
    [48]	valid's binary_logloss: 0.00685491
    [49]	valid's binary_logloss: 0.00681528
    [50]	valid's binary_logloss: 0.00678127
    [51]	valid's binary_logloss: 0.00675512
    [52]	valid's binary_logloss: 0.00672701
    [53]	valid's binary_logloss: 0.00670179
    [54]	valid's binary_logloss: 0.00668305
    [55]	valid's binary_logloss: 0.00666074
    [56]	valid's binary_logloss: 0.00664065
    [57]	valid's binary_logloss: 0.00662354
    [58]	valid's binary_logloss: 0.00660715
    [59]	valid's binary_logloss: 0.00659693
    [60]	valid's binary_logloss: 0.00658394
    [61]	valid's binary_logloss: 0.0065715
    [62]	valid's binary_logloss: 0.00656092
    [63]	valid's binary_logloss: 0.0065514
    [64]	valid's binary_logloss: 0.00654282
    [65]	valid's binary_logloss: 0.0065351
    [66]	valid's binary_logloss: 0.00652813
    [67]	valid's binary_logloss: 0.0065212
    [68]	valid's binary_logloss: 0.00651495
    [69]	valid's binary_logloss: 0.0065093
    [70]	valid's binary_logloss: 0.0065042
    [71]	valid's binary_logloss: 0.0064996
    [72]	valid's binary_logloss: 0.00649544
    [73]	valid's binary_logloss: 0.00649199
    [74]	valid's binary_logloss: 0.00648888
    [75]	valid's binary_logloss: 0.00648607
    [76]	valid's binary_logloss: 0.00648323
    [77]	valid's binary_logloss: 0.00648067
    [78]	valid's binary_logloss: 0.00647859
    [79]	valid's binary_logloss: 0.00647648
    [80]	valid's binary_logloss: 0.00647457
    [81]	valid's binary_logloss: 0.00647285
    [82]	valid's binary_logloss: 0.00647129
    [83]	valid's binary_logloss: 0.00647004
    [84]	valid's binary_logloss: 0.00646875
    [85]	valid's binary_logloss: 0.00646773
    [86]	valid's binary_logloss: 0.0064668
    [87]	valid's binary_logloss: 0.00646596
    [88]	valid's binary_logloss: 0.00646521
    [89]	valid's binary_logloss: 0.00646439
    [90]	valid's binary_logloss: 0.00646366
    [91]	valid's binary_logloss: 0.0064631
    [92]	valid's binary_logloss: 0.0064625
    [93]	valid's binary_logloss: 0.0064625
    [94]	valid's binary_logloss: 0.00646196
    [95]	valid's binary_logloss: 0.00646196
    [96]	valid's binary_logloss: 0.00646147
    [97]	valid's binary_logloss: 0.00646101
    [98]	valid's binary_logloss: 0.00646061
    [99]	valid's binary_logloss: 0.00646061
    [100]	valid's binary_logloss: 0.00646061
    [101]	valid's binary_logloss: 0.00646024
    [102]	valid's binary_logloss: 0.00645991
    [103]	valid's binary_logloss: 0.00645991
    [104]	valid's binary_logloss: 0.0064596
    [105]	valid's binary_logloss: 0.00645933
    [106]	valid's binary_logloss: 0.00645933
    [107]	valid's binary_logloss: 0.00645912
    [108]	valid's binary_logloss: 0.00645893
    [109]	valid's binary_logloss: 0.00645876
    [110]	valid's binary_logloss: 0.0064586
    [111]	valid's binary_logloss: 0.00645842
    [112]	valid's binary_logloss: 0.00645829
    [113]	valid's binary_logloss: 0.00645813
    [114]	valid's binary_logloss: 0.00645799
    [115]	valid's binary_logloss: 0.00645799
    [116]	valid's binary_logloss: 0.00645786
    [117]	valid's binary_logloss: 0.00645786
    [118]	valid's binary_logloss: 0.00645774
    [119]	valid's binary_logloss: 0.00645774
    [120]	valid's binary_logloss: 0.00645774
    [121]	valid's binary_logloss: 0.00645764
    [122]	valid's binary_logloss: 0.00645754
    [123]	valid's binary_logloss: 0.00645746
    [124]	valid's binary_logloss: 0.00645738
    [125]	valid's binary_logloss: 0.00645731
    [126]	valid's binary_logloss: 0.00645724
    [127]	valid's binary_logloss: 0.00645719
    [128]	valid's binary_logloss: 0.00645713
    [129]	valid's binary_logloss: 0.00645709
    [130]	valid's binary_logloss: 0.00645705
    [131]	valid's binary_logloss: 0.00645705
    [132]	valid's binary_logloss: 0.00645705
    [133]	valid's binary_logloss: 0.00645701
    [134]	valid's binary_logloss: 0.00645697
    [135]	valid's binary_logloss: 0.00645694
    [136]	valid's binary_logloss: 0.00645691
    [137]	valid's binary_logloss: 0.00645689
    [138]	valid's binary_logloss: 0.00645686
    [139]	valid's binary_logloss: 0.00645685
    [140]	valid's binary_logloss: 0.00645683
    [141]	valid's binary_logloss: 0.00645682
    [142]	valid's binary_logloss: 0.00645682
    [143]	valid's binary_logloss: 0.0064568
    [144]	valid's binary_logloss: 0.0064568
    [145]	valid's binary_logloss: 0.00645678
    [146]	valid's binary_logloss: 0.00645677
    [147]	valid's binary_logloss: 0.00645676
    [148]	valid's binary_logloss: 0.00645675
    [149]	valid's binary_logloss: 0.00645674
    [150]	valid's binary_logloss: 0.00645674
    [151]	valid's binary_logloss: 0.00645673
    [152]	valid's binary_logloss: 0.00645673
    [153]	valid's binary_logloss: 0.00645672
    [154]	valid's binary_logloss: 0.00645671
    [155]	valid's binary_logloss: 0.00645671
    [156]	valid's binary_logloss: 0.0064567
    [157]	valid's binary_logloss: 0.0064567
    [158]	valid's binary_logloss: 0.00645669
    [159]	valid's binary_logloss: 0.00645669
    [160]	valid's binary_logloss: 0.00645669
    [161]	valid's binary_logloss: 0.00645669
    [162]	valid's binary_logloss: 0.00645669
    [163]	valid's binary_logloss: 0.00645668
    [164]	valid's binary_logloss: 0.00645668
    [165]	valid's binary_logloss: 0.00645668
    [166]	valid's binary_logloss: 0.00645668
    [167]	valid's binary_logloss: 0.00645668
    [168]	valid's binary_logloss: 0.00645668
    [169]	valid's binary_logloss: 0.00645667
    [170]	valid's binary_logloss: 0.00645667
    [171]	valid's binary_logloss: 0.00645667
    [172]	valid's binary_logloss: 0.00645667
    [173]	valid's binary_logloss: 0.00645667
    [174]	valid's binary_logloss: 0.00645667
    [175]	valid's binary_logloss: 0.00645667
    [176]	valid's binary_logloss: 0.00645666
    [177]	valid's binary_logloss: 0.00645666
    [178]	valid's binary_logloss: 0.00645666
    [179]	valid's binary_logloss: 0.00645666
    [180]	valid's binary_logloss: 0.00645666
    [181]	valid's binary_logloss: 0.00645666
    [182]	valid's binary_logloss: 0.00645666
    [183]	valid's binary_logloss: 0.00645666
    [184]	valid's binary_logloss: 0.00645666
    [185]	valid's binary_logloss: 0.00645666
    [186]	valid's binary_logloss: 0.00645666
    [187]	valid's binary_logloss: 0.00645666
    [188]	valid's binary_logloss: 0.00645666
    [189]	valid's binary_logloss: 0.00645665
    [190]	valid's binary_logloss: 0.00645665
    [191]	valid's binary_logloss: 0.00645665
    [192]	valid's binary_logloss: 0.00645665
    [193]	valid's binary_logloss: 0.00645665
    [194]	valid's binary_logloss: 0.00645665
    [195]	valid's binary_logloss: 0.00645665
    [196]	valid's binary_logloss: 0.00645665
    [197]	valid's binary_logloss: 0.00645665
    [198]	valid's binary_logloss: 0.00645665
    [199]	valid's binary_logloss: 0.00645665
    [200]	valid's binary_logloss: 0.00645665
    [201]	valid's binary_logloss: 0.00645665
    [202]	valid's binary_logloss: 0.00645665
    [203]	valid's binary_logloss: 0.00645665
    [204]	valid's binary_logloss: 0.00645665
    [205]	valid's binary_logloss: 0.00645665
    [206]	valid's binary_logloss: 0.00645665
    [207]	valid's binary_logloss: 0.00645665
    [208]	valid's binary_logloss: 0.00645665
    [209]	valid's binary_logloss: 0.00645665
    [210]	valid's binary_logloss: 0.00645665
    [211]	valid's binary_logloss: 0.00645665
    [212]	valid's binary_logloss: 0.00645665
    [213]	valid's binary_logloss: 0.00645665
    [214]	valid's binary_logloss: 0.00645665
    [215]	valid's binary_logloss: 0.00645665
    [216]	valid's binary_logloss: 0.00645665
    [217]	valid's binary_logloss: 0.00645665
    [218]	valid's binary_logloss: 0.00645665
    [219]	valid's binary_logloss: 0.00645665
    [220]	valid's binary_logloss: 0.00645665
    [221]	valid's binary_logloss: 0.00645665
    [222]	valid's binary_logloss: 0.00645665
    [223]	valid's binary_logloss: 0.00645665
    [224]	valid's binary_logloss: 0.00645665
    [225]	valid's binary_logloss: 0.00645665
    [226]	valid's binary_logloss: 0.00645665
    [227]	valid's binary_logloss: 0.00645665
    [228]	valid's binary_logloss: 0.00645665
    [229]	valid's binary_logloss: 0.00645665
    [230]	valid's binary_logloss: 0.00645665
    [231]	valid's binary_logloss: 0.00645664
    [232]	valid's binary_logloss: 0.00645664
    [233]	valid's binary_logloss: 0.00645664
    [234]	valid's binary_logloss: 0.00645664
    [235]	valid's binary_logloss: 0.00645664
    [236]	valid's binary_logloss: 0.00645664
    [237]	valid's binary_logloss: 0.00645664
    [238]	valid's binary_logloss: 0.00645664
    [239]	valid's binary_logloss: 0.00645664
    [240]	valid's binary_logloss: 0.00645664
    [241]	valid's binary_logloss: 0.00645664
    [242]	valid's binary_logloss: 0.00645664
    [243]	valid's binary_logloss: 0.00645664
    [244]	valid's binary_logloss: 0.00645664
    [245]	valid's binary_logloss: 0.00645664
    [246]	valid's binary_logloss: 0.00645664
    [247]	valid's binary_logloss: 0.00645664
    [248]	valid's binary_logloss: 0.00645664
    [249]	valid's binary_logloss: 0.00645664
    [250]	valid's binary_logloss: 0.00645664
    [251]	valid's binary_logloss: 0.00645664
    [252]	valid's binary_logloss: 0.00645664
    [253]	valid's binary_logloss: 0.00645664
    [254]	valid's binary_logloss: 0.00645664
    [255]	valid's binary_logloss: 0.00645664
    [256]	valid's binary_logloss: 0.00645664
    [257]	valid's binary_logloss: 0.00645664
    [258]	valid's binary_logloss: 0.00645664
    [259]	valid's binary_logloss: 0.00645664
    [260]	valid's binary_logloss: 0.00645664
    [261]	valid's binary_logloss: 0.00645664
    [262]	valid's binary_logloss: 0.00645664
    [263]	valid's binary_logloss: 0.00645664
    [264]	valid's binary_logloss: 0.00645664
    [265]	valid's binary_logloss: 0.00645664
    [266]	valid's binary_logloss: 0.00645664
    [267]	valid's binary_logloss: 0.00645664
    [268]	valid's binary_logloss: 0.00645664
    [269]	valid's binary_logloss: 0.00645664
    [270]	valid's binary_logloss: 0.00645664
    [271]	valid's binary_logloss: 0.00645664
    [272]	valid's binary_logloss: 0.00645664
    [273]	valid's binary_logloss: 0.00645664
    [274]	valid's binary_logloss: 0.00645664
    [275]	valid's binary_logloss: 0.00645664
    [276]	valid's binary_logloss: 0.00645664
    [277]	valid's binary_logloss: 0.00645664
    [278]	valid's binary_logloss: 0.00645664
    [279]	valid's binary_logloss: 0.00645664
    [280]	valid's binary_logloss: 0.00645664
    [281]	valid's binary_logloss: 0.00645664
    [282]	valid's binary_logloss: 0.00645664
    [283]	valid's binary_logloss: 0.00645664
    [284]	valid's binary_logloss: 0.00645664
    [285]	valid's binary_logloss: 0.00645664
    [286]	valid's binary_logloss: 0.00645664
    [287]	valid's binary_logloss: 0.00645664
    [288]	valid's binary_logloss: 0.00645664
    [289]	valid's binary_logloss: 0.00645664
    [290]	valid's binary_logloss: 0.00645664
    [291]	valid's binary_logloss: 0.00645664
    [292]	valid's binary_logloss: 0.00645664
    [293]	valid's binary_logloss: 0.00645664
    [294]	valid's binary_logloss: 0.00645664
    [295]	valid's binary_logloss: 0.00645664
    [296]	valid's binary_logloss: 0.00645664
    [297]	valid's binary_logloss: 0.00645664
    [298]	valid's binary_logloss: 0.00645664
    [299]	valid's binary_logloss: 0.00645664
    [300]	valid's binary_logloss: 0.00645664
    [301]	valid's binary_logloss: 0.00645664
    [302]	valid's binary_logloss: 0.00645664
    [303]	valid's binary_logloss: 0.00645664
    [304]	valid's binary_logloss: 0.00645664
    [305]	valid's binary_logloss: 0.00645664
    [306]	valid's binary_logloss: 0.00645664
    [307]	valid's binary_logloss: 0.00645664
    [308]	valid's binary_logloss: 0.00645664
    [309]	valid's binary_logloss: 0.00645664
    [310]	valid's binary_logloss: 0.00645664
    [311]	valid's binary_logloss: 0.00645664
    [312]	valid's binary_logloss: 0.00645664
    [313]	valid's binary_logloss: 0.00645664
    [314]	valid's binary_logloss: 0.00645664
    [315]	valid's binary_logloss: 0.00645664
    [316]	valid's binary_logloss: 0.00645664
    [317]	valid's binary_logloss: 0.00645664
    [318]	valid's binary_logloss: 0.00645664
    [319]	valid's binary_logloss: 0.00645664
    [320]	valid's binary_logloss: 0.00645664
    [321]	valid's binary_logloss: 0.00645664
    [322]	valid's binary_logloss: 0.00645664
    [323]	valid's binary_logloss: 0.00645664
    [324]	valid's binary_logloss: 0.00645664
    [325]	valid's binary_logloss: 0.00645664
    [326]	valid's binary_logloss: 0.00645664
    [327]	valid's binary_logloss: 0.00645664
    [328]	valid's binary_logloss: 0.00645664
    [329]	valid's binary_logloss: 0.00645664
    [330]	valid's binary_logloss: 0.00645664
    [331]	valid's binary_logloss: 0.00645664
    [332]	valid's binary_logloss: 0.00645664
    [333]	valid's binary_logloss: 0.00645664
    [334]	valid's binary_logloss: 0.00645664
    [335]	valid's binary_logloss: 0.00645664
    [336]	valid's binary_logloss: 0.00645664
    [337]	valid's binary_logloss: 0.00645664
    [338]	valid's binary_logloss: 0.00645664
    [339]	valid's binary_logloss: 0.00645664
    [340]	valid's binary_logloss: 0.00645664
    [341]	valid's binary_logloss: 0.00645664
    [342]	valid's binary_logloss: 0.00645664
    [343]	valid's binary_logloss: 0.00645664
    [344]	valid's binary_logloss: 0.00645664
    [345]	valid's binary_logloss: 0.00645664
    [346]	valid's binary_logloss: 0.00645664
    [347]	valid's binary_logloss: 0.00645664
    [348]	valid's binary_logloss: 0.00645664
    [349]	valid's binary_logloss: 0.00645664
    [350]	valid's binary_logloss: 0.00645664
    [351]	valid's binary_logloss: 0.00645664
    [352]	valid's binary_logloss: 0.00645664
    [353]	valid's binary_logloss: 0.00645664
    [354]	valid's binary_logloss: 0.00645664
    [355]	valid's binary_logloss: 0.00645664
    [356]	valid's binary_logloss: 0.00645664
    Early stopping, best iteration is:
    [346]	valid's binary_logloss: 0.00645664
    [1]	valid's binary_logloss: 0.0753199
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.0655947
    [3]	valid's binary_logloss: 0.0578969
    [4]	valid's binary_logloss: 0.0515581
    [5]	valid's binary_logloss: 0.0462527
    [6]	valid's binary_logloss: 0.0417066
    [7]	valid's binary_logloss: 0.0377377
    [8]	valid's binary_logloss: 0.0342827
    [9]	valid's binary_logloss: 0.0312548
    [10]	valid's binary_logloss: 0.0285775
    [11]	valid's binary_logloss: 0.0262117
    [12]	valid's binary_logloss: 0.0241203
    [13]	valid's binary_logloss: 0.0222569
    [14]	valid's binary_logloss: 0.0206046
    [15]	valid's binary_logloss: 0.019151
    [16]	valid's binary_logloss: 0.0178295
    [17]	valid's binary_logloss: 0.0166732
    [18]	valid's binary_logloss: 0.015621
    [19]	valid's binary_logloss: 0.0146739
    [20]	valid's binary_logloss: 0.0138256
    [21]	valid's binary_logloss: 0.0130665
    [22]	valid's binary_logloss: 0.0124079
    [23]	valid's binary_logloss: 0.0117969
    [24]	valid's binary_logloss: 0.011266
    [25]	valid's binary_logloss: 0.0107819
    [26]	valid's binary_logloss: 0.0103405
    [27]	valid's binary_logloss: 0.00994393
    [28]	valid's binary_logloss: 0.00958965
    [29]	valid's binary_logloss: 0.00927006
    [30]	valid's binary_logloss: 0.00900067
    [31]	valid's binary_logloss: 0.00875805
    [32]	valid's binary_logloss: 0.00852717
    [33]	valid's binary_logloss: 0.00833394
    [34]	valid's binary_logloss: 0.00814753
    [35]	valid's binary_logloss: 0.00797745
    [36]	valid's binary_logloss: 0.00782584
    [37]	valid's binary_logloss: 0.00768746
    [38]	valid's binary_logloss: 0.0075633
    [39]	valid's binary_logloss: 0.0074616
    [40]	valid's binary_logloss: 0.00736353
    [41]	valid's binary_logloss: 0.00728194
    [42]	valid's binary_logloss: 0.00720214
    [43]	valid's binary_logloss: 0.00712771
    [44]	valid's binary_logloss: 0.00706353
    [45]	valid's binary_logloss: 0.00700318
    [46]	valid's binary_logloss: 0.00695423
    [47]	valid's binary_logloss: 0.00690507
    [48]	valid's binary_logloss: 0.00686282
    [49]	valid's binary_logloss: 0.00682287
    [50]	valid's binary_logloss: 0.00678871
    [51]	valid's binary_logloss: 0.00676045
    [52]	valid's binary_logloss: 0.00673084
    [53]	valid's binary_logloss: 0.00670555
    [54]	valid's binary_logloss: 0.00668569
    [55]	valid's binary_logloss: 0.00666474
    [56]	valid's binary_logloss: 0.00664469
    [57]	valid's binary_logloss: 0.00662766
    [58]	valid's binary_logloss: 0.00661128
    [59]	valid's binary_logloss: 0.00660004
    [60]	valid's binary_logloss: 0.00658718
    [61]	valid's binary_logloss: 0.00657561
    [62]	valid's binary_logloss: 0.0065652
    [63]	valid's binary_logloss: 0.00655581
    [64]	valid's binary_logloss: 0.00654736
    [65]	valid's binary_logloss: 0.00653975
    [66]	valid's binary_logloss: 0.00653288
    [67]	valid's binary_logloss: 0.00652588
    [68]	valid's binary_logloss: 0.00651956
    [69]	valid's binary_logloss: 0.00651385
    [70]	valid's binary_logloss: 0.0065087
    [71]	valid's binary_logloss: 0.00650404
    [72]	valid's binary_logloss: 0.00649983
    [73]	valid's binary_logloss: 0.00649646
    [74]	valid's binary_logloss: 0.00649341
    [75]	valid's binary_logloss: 0.00649066
    [76]	valid's binary_logloss: 0.00648776
    [77]	valid's binary_logloss: 0.00648515
    [78]	valid's binary_logloss: 0.00648313
    [79]	valid's binary_logloss: 0.00648096
    [80]	valid's binary_logloss: 0.006479
    [81]	valid's binary_logloss: 0.00647723
    [82]	valid's binary_logloss: 0.00647563
    [83]	valid's binary_logloss: 0.00647443
    [84]	valid's binary_logloss: 0.00647314
    [85]	valid's binary_logloss: 0.00647215
    [86]	valid's binary_logloss: 0.00647126
    [87]	valid's binary_logloss: 0.00647045
    [88]	valid's binary_logloss: 0.00646972
    [89]	valid's binary_logloss: 0.00646883
    [90]	valid's binary_logloss: 0.00646802
    [91]	valid's binary_logloss: 0.00646749
    [92]	valid's binary_logloss: 0.0064668
    [93]	valid's binary_logloss: 0.0064668
    [94]	valid's binary_logloss: 0.00646618
    [95]	valid's binary_logloss: 0.00646618
    [96]	valid's binary_logloss: 0.00646562
    [97]	valid's binary_logloss: 0.00646512
    [98]	valid's binary_logloss: 0.00646466
    [99]	valid's binary_logloss: 0.00646466
    [100]	valid's binary_logloss: 0.00646466
    [101]	valid's binary_logloss: 0.00646425
    [102]	valid's binary_logloss: 0.00646387
    [103]	valid's binary_logloss: 0.00646387
    [104]	valid's binary_logloss: 0.00646353
    [105]	valid's binary_logloss: 0.00646323
    [106]	valid's binary_logloss: 0.00646323
    [107]	valid's binary_logloss: 0.00646305
    [108]	valid's binary_logloss: 0.00646289
    [109]	valid's binary_logloss: 0.00646275
    [110]	valid's binary_logloss: 0.00646262
    [111]	valid's binary_logloss: 0.00646239
    [112]	valid's binary_logloss: 0.00646229
    [113]	valid's binary_logloss: 0.00646209
    [114]	valid's binary_logloss: 0.00646191
    [115]	valid's binary_logloss: 0.00646191
    [116]	valid's binary_logloss: 0.00646175
    [117]	valid's binary_logloss: 0.00646175
    [118]	valid's binary_logloss: 0.00646161
    [119]	valid's binary_logloss: 0.00646161
    [120]	valid's binary_logloss: 0.00646161
    [121]	valid's binary_logloss: 0.00646148
    [122]	valid's binary_logloss: 0.00646136
    [123]	valid's binary_logloss: 0.00646125
    [124]	valid's binary_logloss: 0.00646115
    [125]	valid's binary_logloss: 0.00646107
    [126]	valid's binary_logloss: 0.00646099
    [127]	valid's binary_logloss: 0.00646091
    [128]	valid's binary_logloss: 0.00646085
    [129]	valid's binary_logloss: 0.00646079
    [130]	valid's binary_logloss: 0.00646074
    [131]	valid's binary_logloss: 0.00646074
    [132]	valid's binary_logloss: 0.00646074
    [133]	valid's binary_logloss: 0.00646069
    [134]	valid's binary_logloss: 0.00646065
    [135]	valid's binary_logloss: 0.00646061
    [136]	valid's binary_logloss: 0.00646057
    [137]	valid's binary_logloss: 0.00646054
    [138]	valid's binary_logloss: 0.00646051
    [139]	valid's binary_logloss: 0.00646051
    [140]	valid's binary_logloss: 0.00646049
    [141]	valid's binary_logloss: 0.00646046
    [142]	valid's binary_logloss: 0.00646046
    [143]	valid's binary_logloss: 0.00646044
    [144]	valid's binary_logloss: 0.00646044
    [145]	valid's binary_logloss: 0.00646042
    [146]	valid's binary_logloss: 0.00646041
    [147]	valid's binary_logloss: 0.00646039
    [148]	valid's binary_logloss: 0.00646038
    [149]	valid's binary_logloss: 0.00646038
    [150]	valid's binary_logloss: 0.00646036
    [151]	valid's binary_logloss: 0.00646035
    [152]	valid's binary_logloss: 0.00646035
    [153]	valid's binary_logloss: 0.00646034
    [154]	valid's binary_logloss: 0.00646033
    [155]	valid's binary_logloss: 0.00646033
    [156]	valid's binary_logloss: 0.00646032
    [157]	valid's binary_logloss: 0.00646032
    [158]	valid's binary_logloss: 0.00646031
    [159]	valid's binary_logloss: 0.00646031
    [160]	valid's binary_logloss: 0.00646031
    [161]	valid's binary_logloss: 0.00646031
    [162]	valid's binary_logloss: 0.0064603
    [163]	valid's binary_logloss: 0.0064603
    [164]	valid's binary_logloss: 0.00646029
    [165]	valid's binary_logloss: 0.00646029
    [166]	valid's binary_logloss: 0.00646029
    [167]	valid's binary_logloss: 0.00646029
    [168]	valid's binary_logloss: 0.00646029
    [169]	valid's binary_logloss: 0.00646028
    [170]	valid's binary_logloss: 0.00646028
    [171]	valid's binary_logloss: 0.00646028
    [172]	valid's binary_logloss: 0.00646027
    [173]	valid's binary_logloss: 0.00646027
    [174]	valid's binary_logloss: 0.00646027
    [175]	valid's binary_logloss: 0.00646027
    [176]	valid's binary_logloss: 0.00646027
    [177]	valid's binary_logloss: 0.00646027
    [178]	valid's binary_logloss: 0.00646027
    [179]	valid's binary_logloss: 0.00646026
    [180]	valid's binary_logloss: 0.00646026
    [181]	valid's binary_logloss: 0.00646026
    [182]	valid's binary_logloss: 0.00646026
    [183]	valid's binary_logloss: 0.00646026
    [184]	valid's binary_logloss: 0.00646026
    [185]	valid's binary_logloss: 0.00646026
    [186]	valid's binary_logloss: 0.00646026
    [187]	valid's binary_logloss: 0.00646026
    [188]	valid's binary_logloss: 0.00646026
    [189]	valid's binary_logloss: 0.00646026
    [190]	valid's binary_logloss: 0.00646025
    [191]	valid's binary_logloss: 0.00646025
    [192]	valid's binary_logloss: 0.00646025
    [193]	valid's binary_logloss: 0.00646025
    [194]	valid's binary_logloss: 0.00646025
    [195]	valid's binary_logloss: 0.00646025
    [196]	valid's binary_logloss: 0.00646025
    [197]	valid's binary_logloss: 0.00646025
    [198]	valid's binary_logloss: 0.00646025
    [199]	valid's binary_logloss: 0.00646025
    [200]	valid's binary_logloss: 0.00646025
    [201]	valid's binary_logloss: 0.00646025
    [202]	valid's binary_logloss: 0.00646025
    [203]	valid's binary_logloss: 0.00646025
    [204]	valid's binary_logloss: 0.00646025
    [205]	valid's binary_logloss: 0.00646025
    [206]	valid's binary_logloss: 0.00646025
    [207]	valid's binary_logloss: 0.00646025
    [208]	valid's binary_logloss: 0.00646025
    [209]	valid's binary_logloss: 0.00646025
    [210]	valid's binary_logloss: 0.00646025
    [211]	valid's binary_logloss: 0.00646025
    [212]	valid's binary_logloss: 0.00646025
    [213]	valid's binary_logloss: 0.00646025
    [214]	valid's binary_logloss: 0.00646025
    [215]	valid's binary_logloss: 0.00646025
    [216]	valid's binary_logloss: 0.00646025
    [217]	valid's binary_logloss: 0.00646025
    [218]	valid's binary_logloss: 0.00646025
    [219]	valid's binary_logloss: 0.00646025
    [220]	valid's binary_logloss: 0.00646024
    [221]	valid's binary_logloss: 0.00646024
    [222]	valid's binary_logloss: 0.00646024
    [223]	valid's binary_logloss: 0.00646024
    [224]	valid's binary_logloss: 0.00646024
    [225]	valid's binary_logloss: 0.00646024
    [226]	valid's binary_logloss: 0.00646024
    [227]	valid's binary_logloss: 0.00646024
    [228]	valid's binary_logloss: 0.00646024
    [229]	valid's binary_logloss: 0.00646024
    [230]	valid's binary_logloss: 0.00646024
    [231]	valid's binary_logloss: 0.00646024
    [232]	valid's binary_logloss: 0.00646024
    [233]	valid's binary_logloss: 0.00646024
    [234]	valid's binary_logloss: 0.00646024
    [235]	valid's binary_logloss: 0.00646024
    [236]	valid's binary_logloss: 0.00646024
    [237]	valid's binary_logloss: 0.00646024
    [238]	valid's binary_logloss: 0.00646024
    [239]	valid's binary_logloss: 0.00646024
    [240]	valid's binary_logloss: 0.00646024
    [241]	valid's binary_logloss: 0.00646024
    [242]	valid's binary_logloss: 0.00646024
    [243]	valid's binary_logloss: 0.00646024
    [244]	valid's binary_logloss: 0.00646024
    [245]	valid's binary_logloss: 0.00646024
    [246]	valid's binary_logloss: 0.00646024
    [247]	valid's binary_logloss: 0.00646024
    [248]	valid's binary_logloss: 0.00646024
    [249]	valid's binary_logloss: 0.00646024
    [250]	valid's binary_logloss: 0.00646024
    [251]	valid's binary_logloss: 0.00646024
    [252]	valid's binary_logloss: 0.00646024
    [253]	valid's binary_logloss: 0.00646024
    [254]	valid's binary_logloss: 0.00646024
    [255]	valid's binary_logloss: 0.00646024
    [256]	valid's binary_logloss: 0.00646024
    [257]	valid's binary_logloss: 0.00646024
    [258]	valid's binary_logloss: 0.00646024
    [259]	valid's binary_logloss: 0.00646024
    [260]	valid's binary_logloss: 0.00646024
    [261]	valid's binary_logloss: 0.00646024
    [262]	valid's binary_logloss: 0.00646024
    [263]	valid's binary_logloss: 0.00646024
    [264]	valid's binary_logloss: 0.00646024
    [265]	valid's binary_logloss: 0.00646024
    [266]	valid's binary_logloss: 0.00646024
    [267]	valid's binary_logloss: 0.00646024
    [268]	valid's binary_logloss: 0.00646024
    [269]	valid's binary_logloss: 0.00646024
    [270]	valid's binary_logloss: 0.00646024
    [271]	valid's binary_logloss: 0.00646024
    [272]	valid's binary_logloss: 0.00646024
    [273]	valid's binary_logloss: 0.00646024
    [274]	valid's binary_logloss: 0.00646024
    [275]	valid's binary_logloss: 0.00646024
    [276]	valid's binary_logloss: 0.00646024
    [277]	valid's binary_logloss: 0.00646024
    [278]	valid's binary_logloss: 0.00646024
    [279]	valid's binary_logloss: 0.00646024
    [280]	valid's binary_logloss: 0.00646024
    [281]	valid's binary_logloss: 0.00646024
    [282]	valid's binary_logloss: 0.00646024
    [283]	valid's binary_logloss: 0.00646024
    [284]	valid's binary_logloss: 0.00646024
    [285]	valid's binary_logloss: 0.00646024
    [286]	valid's binary_logloss: 0.00646024
    [287]	valid's binary_logloss: 0.00646024
    [288]	valid's binary_logloss: 0.00646024
    [289]	valid's binary_logloss: 0.00646024
    [290]	valid's binary_logloss: 0.00646024
    [291]	valid's binary_logloss: 0.00646024
    [292]	valid's binary_logloss: 0.00646024
    [293]	valid's binary_logloss: 0.00646024
    [294]	valid's binary_logloss: 0.00646024
    [295]	valid's binary_logloss: 0.00646024
    [296]	valid's binary_logloss: 0.00646024
    [297]	valid's binary_logloss: 0.00646024
    [298]	valid's binary_logloss: 0.00646024
    [299]	valid's binary_logloss: 0.00646024
    [300]	valid's binary_logloss: 0.00646024
    [301]	valid's binary_logloss: 0.00646024
    [302]	valid's binary_logloss: 0.00646024
    [303]	valid's binary_logloss: 0.00646024
    [304]	valid's binary_logloss: 0.00646024
    [305]	valid's binary_logloss: 0.00646024
    [306]	valid's binary_logloss: 0.00646024
    [307]	valid's binary_logloss: 0.00646024
    [308]	valid's binary_logloss: 0.00646024
    [309]	valid's binary_logloss: 0.00646024
    [310]	valid's binary_logloss: 0.00646024
    [311]	valid's binary_logloss: 0.00646024
    [312]	valid's binary_logloss: 0.00646024
    [313]	valid's binary_logloss: 0.00646024
    [314]	valid's binary_logloss: 0.00646024
    [315]	valid's binary_logloss: 0.00646024
    [316]	valid's binary_logloss: 0.00646024
    [317]	valid's binary_logloss: 0.00646024
    [318]	valid's binary_logloss: 0.00646024
    [319]	valid's binary_logloss: 0.00646024
    [320]	valid's binary_logloss: 0.00646024
    [321]	valid's binary_logloss: 0.00646024
    [322]	valid's binary_logloss: 0.00646024
    [323]	valid's binary_logloss: 0.00646024
    [324]	valid's binary_logloss: 0.00646024
    [325]	valid's binary_logloss: 0.00646024
    [326]	valid's binary_logloss: 0.00646024
    [327]	valid's binary_logloss: 0.00646024
    [328]	valid's binary_logloss: 0.00646024
    [329]	valid's binary_logloss: 0.00646024
    [330]	valid's binary_logloss: 0.00646024
    [331]	valid's binary_logloss: 0.00646024
    [332]	valid's binary_logloss: 0.00646024
    [333]	valid's binary_logloss: 0.00646024
    [334]	valid's binary_logloss: 0.00646024
    [335]	valid's binary_logloss: 0.00646024
    [336]	valid's binary_logloss: 0.00646024
    [337]	valid's binary_logloss: 0.00646024
    [338]	valid's binary_logloss: 0.00646024
    [339]	valid's binary_logloss: 0.00646024
    [340]	valid's binary_logloss: 0.00646024
    [341]	valid's binary_logloss: 0.00646024
    [342]	valid's binary_logloss: 0.00646024
    [343]	valid's binary_logloss: 0.00646024
    [344]	valid's binary_logloss: 0.00646024
    [345]	valid's binary_logloss: 0.00646024
    [346]	valid's binary_logloss: 0.00646024
    [347]	valid's binary_logloss: 0.00646024
    [348]	valid's binary_logloss: 0.00646024
    [349]	valid's binary_logloss: 0.00646024
    [350]	valid's binary_logloss: 0.00646024
    [351]	valid's binary_logloss: 0.00646024
    [352]	valid's binary_logloss: 0.00646024
    [353]	valid's binary_logloss: 0.00646024
    [354]	valid's binary_logloss: 0.00646024
    [355]	valid's binary_logloss: 0.00646024
    [356]	valid's binary_logloss: 0.00646024
    [357]	valid's binary_logloss: 0.00646024
    [358]	valid's binary_logloss: 0.00646024
    [359]	valid's binary_logloss: 0.00646024
    [360]	valid's binary_logloss: 0.00646024
    [361]	valid's binary_logloss: 0.00646024
    [362]	valid's binary_logloss: 0.00646024
    [363]	valid's binary_logloss: 0.00646024
    [364]	valid's binary_logloss: 0.00646024
    [365]	valid's binary_logloss: 0.00646024
    [366]	valid's binary_logloss: 0.00646024
    [367]	valid's binary_logloss: 0.00646024
    [368]	valid's binary_logloss: 0.00646024
    [369]	valid's binary_logloss: 0.00646024
    [370]	valid's binary_logloss: 0.00646024
    [371]	valid's binary_logloss: 0.00646024
    [372]	valid's binary_logloss: 0.00646024
    [373]	valid's binary_logloss: 0.00646024
    [374]	valid's binary_logloss: 0.00646024
    [375]	valid's binary_logloss: 0.00646024
    [376]	valid's binary_logloss: 0.00646024
    [377]	valid's binary_logloss: 0.00646024
    [378]	valid's binary_logloss: 0.00646024
    [379]	valid's binary_logloss: 0.00646024
    [380]	valid's binary_logloss: 0.00646024
    [381]	valid's binary_logloss: 0.00646024
    [382]	valid's binary_logloss: 0.00646024
    [383]	valid's binary_logloss: 0.00646024
    [384]	valid's binary_logloss: 0.00646024
    [385]	valid's binary_logloss: 0.00646024
    [386]	valid's binary_logloss: 0.00646024
    [387]	valid's binary_logloss: 0.00646024
    [388]	valid's binary_logloss: 0.00646024
    [389]	valid's binary_logloss: 0.00646024
    [390]	valid's binary_logloss: 0.00646024
    [391]	valid's binary_logloss: 0.00646024
    [392]	valid's binary_logloss: 0.00646024
    [393]	valid's binary_logloss: 0.00646024
    [394]	valid's binary_logloss: 0.00646024
    [395]	valid's binary_logloss: 0.00646024
    [396]	valid's binary_logloss: 0.00646024
    [397]	valid's binary_logloss: 0.00646024
    [398]	valid's binary_logloss: 0.00646024
    [399]	valid's binary_logloss: 0.00646024
    [400]	valid's binary_logloss: 0.00646024
    [401]	valid's binary_logloss: 0.00646024
    [402]	valid's binary_logloss: 0.00646024
    [403]	valid's binary_logloss: 0.00646024
    [404]	valid's binary_logloss: 0.00646024
    [405]	valid's binary_logloss: 0.00646024
    [406]	valid's binary_logloss: 0.00646024
    [407]	valid's binary_logloss: 0.00646024
    [408]	valid's binary_logloss: 0.00646024
    [409]	valid's binary_logloss: 0.00646024
    [410]	valid's binary_logloss: 0.00646024
    [411]	valid's binary_logloss: 0.00646024
    [412]	valid's binary_logloss: 0.00646024
    [413]	valid's binary_logloss: 0.00646024
    [414]	valid's binary_logloss: 0.00646024
    [415]	valid's binary_logloss: 0.00646024
    [416]	valid's binary_logloss: 0.00646024
    [417]	valid's binary_logloss: 0.00646024
    [418]	valid's binary_logloss: 0.00646024
    [419]	valid's binary_logloss: 0.00646024
    [420]	valid's binary_logloss: 0.00646024
    [421]	valid's binary_logloss: 0.00646024
    [422]	valid's binary_logloss: 0.00646024
    [423]	valid's binary_logloss: 0.00646024
    [424]	valid's binary_logloss: 0.00646024
    [425]	valid's binary_logloss: 0.00646024
    [426]	valid's binary_logloss: 0.00646024
    [427]	valid's binary_logloss: 0.00646024
    [428]	valid's binary_logloss: 0.00646024
    [429]	valid's binary_logloss: 0.00646024
    [430]	valid's binary_logloss: 0.00646024
    [431]	valid's binary_logloss: 0.00646024
    [432]	valid's binary_logloss: 0.00646024
    [433]	valid's binary_logloss: 0.00646024
    [434]	valid's binary_logloss: 0.00646024
    [435]	valid's binary_logloss: 0.00646024
    [436]	valid's binary_logloss: 0.00646024
    [437]	valid's binary_logloss: 0.00646024
    [438]	valid's binary_logloss: 0.00646024
    [439]	valid's binary_logloss: 0.00646024
    [440]	valid's binary_logloss: 0.00646024
    [441]	valid's binary_logloss: 0.00646024
    [442]	valid's binary_logloss: 0.00646024
    [443]	valid's binary_logloss: 0.00646024
    [444]	valid's binary_logloss: 0.00646024
    [445]	valid's binary_logloss: 0.00646024
    [446]	valid's binary_logloss: 0.00646024
    [447]	valid's binary_logloss: 0.00646024
    [448]	valid's binary_logloss: 0.00646024
    [449]	valid's binary_logloss: 0.00646024
    [450]	valid's binary_logloss: 0.00646024
    [451]	valid's binary_logloss: 0.00646024
    [452]	valid's binary_logloss: 0.00646024
    [453]	valid's binary_logloss: 0.00646024
    [454]	valid's binary_logloss: 0.00646024
    [455]	valid's binary_logloss: 0.00646024
    [456]	valid's binary_logloss: 0.00646024
    [457]	valid's binary_logloss: 0.00646024
    [458]	valid's binary_logloss: 0.00646024
    [459]	valid's binary_logloss: 0.00646024
    [460]	valid's binary_logloss: 0.00646024
    [461]	valid's binary_logloss: 0.00646024
    [462]	valid's binary_logloss: 0.00646024
    [463]	valid's binary_logloss: 0.00646024
    [464]	valid's binary_logloss: 0.00646024
    [465]	valid's binary_logloss: 0.00646024
    [466]	valid's binary_logloss: 0.00646024
    [467]	valid's binary_logloss: 0.00646024
    [468]	valid's binary_logloss: 0.00646024
    [469]	valid's binary_logloss: 0.00646024
    [470]	valid's binary_logloss: 0.00646024
    [471]	valid's binary_logloss: 0.00646024
    [472]	valid's binary_logloss: 0.00646024
    [473]	valid's binary_logloss: 0.00646024
    [474]	valid's binary_logloss: 0.00646024
    [475]	valid's binary_logloss: 0.00646024
    [476]	valid's binary_logloss: 0.00646024
    [477]	valid's binary_logloss: 0.00646024
    [478]	valid's binary_logloss: 0.00646024
    [479]	valid's binary_logloss: 0.00646024
    [480]	valid's binary_logloss: 0.00646024
    [481]	valid's binary_logloss: 0.00646024
    [482]	valid's binary_logloss: 0.00646024
    [483]	valid's binary_logloss: 0.00646024
    [484]	valid's binary_logloss: 0.00646024
    [485]	valid's binary_logloss: 0.00646024
    [486]	valid's binary_logloss: 0.00646024
    [487]	valid's binary_logloss: 0.00646024
    [488]	valid's binary_logloss: 0.00646024
    [489]	valid's binary_logloss: 0.00646024
    [490]	valid's binary_logloss: 0.00646024
    [491]	valid's binary_logloss: 0.00646024
    [492]	valid's binary_logloss: 0.00646024
    [493]	valid's binary_logloss: 0.00646024
    [494]	valid's binary_logloss: 0.00646024
    [495]	valid's binary_logloss: 0.00646024
    [496]	valid's binary_logloss: 0.00646024
    [497]	valid's binary_logloss: 0.00646024
    [498]	valid's binary_logloss: 0.00646024
    [499]	valid's binary_logloss: 0.00646024
    [500]	valid's binary_logloss: 0.00646024
    [501]	valid's binary_logloss: 0.00646024
    [502]	valid's binary_logloss: 0.00646024
    [503]	valid's binary_logloss: 0.00646024
    [504]	valid's binary_logloss: 0.00646024
    [505]	valid's binary_logloss: 0.00646024
    [506]	valid's binary_logloss: 0.00646024
    [507]	valid's binary_logloss: 0.00646024
    [508]	valid's binary_logloss: 0.00646024
    [509]	valid's binary_logloss: 0.00646024
    [510]	valid's binary_logloss: 0.00646024
    [511]	valid's binary_logloss: 0.00646024
    [512]	valid's binary_logloss: 0.00646024
    [513]	valid's binary_logloss: 0.00646024
    [514]	valid's binary_logloss: 0.00646024
    [515]	valid's binary_logloss: 0.00646024
    [516]	valid's binary_logloss: 0.00646024
    [517]	valid's binary_logloss: 0.00646024
    [518]	valid's binary_logloss: 0.00646024
    [519]	valid's binary_logloss: 0.00646024
    [520]	valid's binary_logloss: 0.00646024
    [521]	valid's binary_logloss: 0.00646024
    [522]	valid's binary_logloss: 0.00646024
    [523]	valid's binary_logloss: 0.00646024
    [524]	valid's binary_logloss: 0.00646024
    [525]	valid's binary_logloss: 0.00646024
    [526]	valid's binary_logloss: 0.00646024
    [527]	valid's binary_logloss: 0.00646024
    [528]	valid's binary_logloss: 0.00646024
    [529]	valid's binary_logloss: 0.00646024
    [530]	valid's binary_logloss: 0.00646024
    [531]	valid's binary_logloss: 0.00646024
    [532]	valid's binary_logloss: 0.00646024
    [533]	valid's binary_logloss: 0.00646024
    [534]	valid's binary_logloss: 0.00646024
    [535]	valid's binary_logloss: 0.00646024
    [536]	valid's binary_logloss: 0.00646024
    Early stopping, best iteration is:
    [526]	valid's binary_logloss: 0.00646024
    [1]	valid's binary_logloss: 0.155808
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.154872
    [3]	valid's binary_logloss: 0.153965
    [4]	valid's binary_logloss: 0.153073
    [5]	valid's binary_logloss: 0.152204
    [6]	valid's binary_logloss: 0.151353
    [7]	valid's binary_logloss: 0.150521
    [8]	valid's binary_logloss: 0.149708
    [9]	valid's binary_logloss: 0.148912
    [10]	valid's binary_logloss: 0.148133
    [11]	valid's binary_logloss: 0.14737
    [12]	valid's binary_logloss: 0.146622
    [13]	valid's binary_logloss: 0.145889
    [14]	valid's binary_logloss: 0.14517
    [15]	valid's binary_logloss: 0.144465
    [16]	valid's binary_logloss: 0.143774
    [17]	valid's binary_logloss: 0.143094
    [18]	valid's binary_logloss: 0.142426
    [19]	valid's binary_logloss: 0.14177
    [20]	valid's binary_logloss: 0.141126
    [21]	valid's binary_logloss: 0.140492
    [22]	valid's binary_logloss: 0.139869
    [23]	valid's binary_logloss: 0.139259
    [24]	valid's binary_logloss: 0.138661
    [25]	valid's binary_logloss: 0.138068
    [26]	valid's binary_logloss: 0.137483
    [27]	valid's binary_logloss: 0.136908
    [28]	valid's binary_logloss: 0.136341
    [29]	valid's binary_logloss: 0.135785
    [30]	valid's binary_logloss: 0.135234
    [31]	valid's binary_logloss: 0.134692
    [32]	valid's binary_logloss: 0.134157
    [33]	valid's binary_logloss: 0.133629
    [34]	valid's binary_logloss: 0.13311
    [35]	valid's binary_logloss: 0.132598
    [36]	valid's binary_logloss: 0.132093
    [37]	valid's binary_logloss: 0.131593
    [38]	valid's binary_logloss: 0.1311
    [39]	valid's binary_logloss: 0.130613
    [40]	valid's binary_logloss: 0.130133
    [41]	valid's binary_logloss: 0.129658
    [42]	valid's binary_logloss: 0.12919
    [43]	valid's binary_logloss: 0.128726
    [44]	valid's binary_logloss: 0.12827
    [45]	valid's binary_logloss: 0.127818
    [46]	valid's binary_logloss: 0.127371
    [47]	valid's binary_logloss: 0.126929
    [48]	valid's binary_logloss: 0.126492
    [49]	valid's binary_logloss: 0.126061
    [50]	valid's binary_logloss: 0.125634
    [51]	valid's binary_logloss: 0.125211
    [52]	valid's binary_logloss: 0.124794
    [53]	valid's binary_logloss: 0.12438
    [54]	valid's binary_logloss: 0.123971
    [55]	valid's binary_logloss: 0.123567
    [56]	valid's binary_logloss: 0.123168
    [57]	valid's binary_logloss: 0.122773
    [58]	valid's binary_logloss: 0.122381
    [59]	valid's binary_logloss: 0.121993
    [60]	valid's binary_logloss: 0.121609
    [61]	valid's binary_logloss: 0.121229
    [62]	valid's binary_logloss: 0.120852
    [63]	valid's binary_logloss: 0.120479
    [64]	valid's binary_logloss: 0.12011
    [65]	valid's binary_logloss: 0.119744
    [66]	valid's binary_logloss: 0.119381
    [67]	valid's binary_logloss: 0.119022
    [68]	valid's binary_logloss: 0.118666
    [69]	valid's binary_logloss: 0.118314
    [70]	valid's binary_logloss: 0.117964
    [71]	valid's binary_logloss: 0.117618
    [72]	valid's binary_logloss: 0.117274
    [73]	valid's binary_logloss: 0.116934
    [74]	valid's binary_logloss: 0.116597
    [75]	valid's binary_logloss: 0.116262
    [76]	valid's binary_logloss: 0.115931
    [77]	valid's binary_logloss: 0.115602
    [78]	valid's binary_logloss: 0.115277
    [79]	valid's binary_logloss: 0.114954
    [80]	valid's binary_logloss: 0.114633
    [81]	valid's binary_logloss: 0.114315
    [82]	valid's binary_logloss: 0.113999
    [83]	valid's binary_logloss: 0.113686
    [84]	valid's binary_logloss: 0.113376
    [85]	valid's binary_logloss: 0.113068
    [86]	valid's binary_logloss: 0.112762
    [87]	valid's binary_logloss: 0.112459
    [88]	valid's binary_logloss: 0.112159
    [89]	valid's binary_logloss: 0.11186
    [90]	valid's binary_logloss: 0.111564
    [91]	valid's binary_logloss: 0.11127
    [92]	valid's binary_logloss: 0.110978
    [93]	valid's binary_logloss: 0.110688
    [94]	valid's binary_logloss: 0.110401
    [95]	valid's binary_logloss: 0.110115
    [96]	valid's binary_logloss: 0.109831
    [97]	valid's binary_logloss: 0.10955
    [98]	valid's binary_logloss: 0.10927
    [99]	valid's binary_logloss: 0.108993
    [100]	valid's binary_logloss: 0.108717
    Did not meet early stopping. Best iteration is:
    [100]	valid's binary_logloss: 0.108717
    [1]	valid's binary_logloss: 0.155805
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.154868
    [3]	valid's binary_logloss: 0.153958
    [4]	valid's binary_logloss: 0.153063
    [5]	valid's binary_logloss: 0.152193
    [6]	valid's binary_logloss: 0.15134
    [7]	valid's binary_logloss: 0.150506
    [8]	valid's binary_logloss: 0.14969
    [9]	valid's binary_logloss: 0.148893
    [10]	valid's binary_logloss: 0.148112
    [11]	valid's binary_logloss: 0.147348
    [12]	valid's binary_logloss: 0.146599
    [13]	valid's binary_logloss: 0.145865
    [14]	valid's binary_logloss: 0.145144
    [15]	valid's binary_logloss: 0.144438
    [16]	valid's binary_logloss: 0.143746
    [17]	valid's binary_logloss: 0.143065
    [18]	valid's binary_logloss: 0.142396
    [19]	valid's binary_logloss: 0.141739
    [20]	valid's binary_logloss: 0.141094
    [21]	valid's binary_logloss: 0.14046
    [22]	valid's binary_logloss: 0.139837
    [23]	valid's binary_logloss: 0.139225
    [24]	valid's binary_logloss: 0.138625
    [25]	valid's binary_logloss: 0.138032
    [26]	valid's binary_logloss: 0.137446
    [27]	valid's binary_logloss: 0.13687
    [28]	valid's binary_logloss: 0.136303
    [29]	valid's binary_logloss: 0.135746
    [30]	valid's binary_logloss: 0.135194
    [31]	valid's binary_logloss: 0.13465
    [32]	valid's binary_logloss: 0.134114
    [33]	valid's binary_logloss: 0.133587
    [34]	valid's binary_logloss: 0.133067
    [35]	valid's binary_logloss: 0.132554
    [36]	valid's binary_logloss: 0.132048
    [37]	valid's binary_logloss: 0.131547
    [38]	valid's binary_logloss: 0.131054
    [39]	valid's binary_logloss: 0.130567
    [40]	valid's binary_logloss: 0.130086
    [41]	valid's binary_logloss: 0.129611
    [42]	valid's binary_logloss: 0.129142
    [43]	valid's binary_logloss: 0.128678
    [44]	valid's binary_logloss: 0.128221
    [45]	valid's binary_logloss: 0.127769
    [46]	valid's binary_logloss: 0.127322
    [47]	valid's binary_logloss: 0.126879
    [48]	valid's binary_logloss: 0.126442
    [49]	valid's binary_logloss: 0.12601
    [50]	valid's binary_logloss: 0.125583
    [51]	valid's binary_logloss: 0.12516
    [52]	valid's binary_logloss: 0.124741
    [53]	valid's binary_logloss: 0.124327
    [54]	valid's binary_logloss: 0.123918
    [55]	valid's binary_logloss: 0.123514
    [56]	valid's binary_logloss: 0.123114
    [57]	valid's binary_logloss: 0.122718
    [58]	valid's binary_logloss: 0.122325
    [59]	valid's binary_logloss: 0.121937
    [60]	valid's binary_logloss: 0.121552
    [61]	valid's binary_logloss: 0.121172
    [62]	valid's binary_logloss: 0.120795
    [63]	valid's binary_logloss: 0.120422
    [64]	valid's binary_logloss: 0.120052
    [65]	valid's binary_logloss: 0.119686
    [66]	valid's binary_logloss: 0.119323
    [67]	valid's binary_logloss: 0.118964
    [68]	valid's binary_logloss: 0.118608
    [69]	valid's binary_logloss: 0.118255
    [70]	valid's binary_logloss: 0.117906
    [71]	valid's binary_logloss: 0.117559
    [72]	valid's binary_logloss: 0.117215
    [73]	valid's binary_logloss: 0.116875
    [74]	valid's binary_logloss: 0.116537
    [75]	valid's binary_logloss: 0.116203
    [76]	valid's binary_logloss: 0.115871
    [77]	valid's binary_logloss: 0.115543
    [78]	valid's binary_logloss: 0.115218
    [79]	valid's binary_logloss: 0.114894
    [80]	valid's binary_logloss: 0.114573
    [81]	valid's binary_logloss: 0.114254
    [82]	valid's binary_logloss: 0.113938
    [83]	valid's binary_logloss: 0.113625
    [84]	valid's binary_logloss: 0.113314
    [85]	valid's binary_logloss: 0.113006
    [86]	valid's binary_logloss: 0.1127
    [87]	valid's binary_logloss: 0.112396
    [88]	valid's binary_logloss: 0.112096
    [89]	valid's binary_logloss: 0.111797
    [90]	valid's binary_logloss: 0.1115
    [91]	valid's binary_logloss: 0.111206
    [92]	valid's binary_logloss: 0.110914
    [93]	valid's binary_logloss: 0.110624
    [94]	valid's binary_logloss: 0.110336
    [95]	valid's binary_logloss: 0.110051
    [96]	valid's binary_logloss: 0.109767
    [97]	valid's binary_logloss: 0.109485
    [98]	valid's binary_logloss: 0.109206
    [99]	valid's binary_logloss: 0.108928
    [100]	valid's binary_logloss: 0.108652
    Did not meet early stopping. Best iteration is:
    [100]	valid's binary_logloss: 0.108652
    [1]	valid's binary_logloss: 0.155806
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.15487
    [3]	valid's binary_logloss: 0.15396
    [4]	valid's binary_logloss: 0.153068
    [5]	valid's binary_logloss: 0.152198
    [6]	valid's binary_logloss: 0.151347
    [7]	valid's binary_logloss: 0.150515
    [8]	valid's binary_logloss: 0.149701
    [9]	valid's binary_logloss: 0.148904
    [10]	valid's binary_logloss: 0.148125
    [11]	valid's binary_logloss: 0.147361
    [12]	valid's binary_logloss: 0.146613
    [13]	valid's binary_logloss: 0.145879
    [14]	valid's binary_logloss: 0.14516
    [15]	valid's binary_logloss: 0.144454
    [16]	valid's binary_logloss: 0.143763
    [17]	valid's binary_logloss: 0.143082
    [18]	valid's binary_logloss: 0.142415
    [19]	valid's binary_logloss: 0.141758
    [20]	valid's binary_logloss: 0.141113
    [21]	valid's binary_logloss: 0.140479
    [22]	valid's binary_logloss: 0.139856
    [23]	valid's binary_logloss: 0.139244
    [24]	valid's binary_logloss: 0.138645
    [25]	valid's binary_logloss: 0.138052
    [26]	valid's binary_logloss: 0.137467
    [27]	valid's binary_logloss: 0.136892
    [28]	valid's binary_logloss: 0.136325
    [29]	valid's binary_logloss: 0.135768
    [30]	valid's binary_logloss: 0.135217
    [31]	valid's binary_logloss: 0.134674
    [32]	valid's binary_logloss: 0.134139
    [33]	valid's binary_logloss: 0.133611
    [34]	valid's binary_logloss: 0.133092
    [35]	valid's binary_logloss: 0.13258
    [36]	valid's binary_logloss: 0.132074
    [37]	valid's binary_logloss: 0.131574
    [38]	valid's binary_logloss: 0.131082
    [39]	valid's binary_logloss: 0.130595
    [40]	valid's binary_logloss: 0.130114
    [41]	valid's binary_logloss: 0.129639
    [42]	valid's binary_logloss: 0.129171
    [43]	valid's binary_logloss: 0.128707
    [44]	valid's binary_logloss: 0.12825
    [45]	valid's binary_logloss: 0.127799
    [46]	valid's binary_logloss: 0.127351
    [47]	valid's binary_logloss: 0.126909
    [48]	valid's binary_logloss: 0.126472
    [49]	valid's binary_logloss: 0.12604
    [50]	valid's binary_logloss: 0.125614
    [51]	valid's binary_logloss: 0.125191
    [52]	valid's binary_logloss: 0.124773
    [53]	valid's binary_logloss: 0.12436
    [54]	valid's binary_logloss: 0.123951
    [55]	valid's binary_logloss: 0.123547
    [56]	valid's binary_logloss: 0.123147
    [57]	valid's binary_logloss: 0.122751
    [58]	valid's binary_logloss: 0.122359
    [59]	valid's binary_logloss: 0.121971
    [60]	valid's binary_logloss: 0.121587
    [61]	valid's binary_logloss: 0.121207
    [62]	valid's binary_logloss: 0.12083
    [63]	valid's binary_logloss: 0.120457
    [64]	valid's binary_logloss: 0.120088
    [65]	valid's binary_logloss: 0.119721
    [66]	valid's binary_logloss: 0.119359
    [67]	valid's binary_logloss: 0.119
    [68]	valid's binary_logloss: 0.118644
    [69]	valid's binary_logloss: 0.118291
    [70]	valid's binary_logloss: 0.117942
    [71]	valid's binary_logloss: 0.117595
    [72]	valid's binary_logloss: 0.117252
    [73]	valid's binary_logloss: 0.116912
    [74]	valid's binary_logloss: 0.116574
    [75]	valid's binary_logloss: 0.11624
    [76]	valid's binary_logloss: 0.115908
    [77]	valid's binary_logloss: 0.11558
    [78]	valid's binary_logloss: 0.115255
    [79]	valid's binary_logloss: 0.114931
    [80]	valid's binary_logloss: 0.11461
    [81]	valid's binary_logloss: 0.114292
    [82]	valid's binary_logloss: 0.113977
    [83]	valid's binary_logloss: 0.113664
    [84]	valid's binary_logloss: 0.113353
    [85]	valid's binary_logloss: 0.113045
    [86]	valid's binary_logloss: 0.112739
    [87]	valid's binary_logloss: 0.112436
    [88]	valid's binary_logloss: 0.112135
    [89]	valid's binary_logloss: 0.111837
    [90]	valid's binary_logloss: 0.111541
    [91]	valid's binary_logloss: 0.111247
    [92]	valid's binary_logloss: 0.110955
    [93]	valid's binary_logloss: 0.110665
    [94]	valid's binary_logloss: 0.110378
    [95]	valid's binary_logloss: 0.110092
    [96]	valid's binary_logloss: 0.109809
    [97]	valid's binary_logloss: 0.109527
    [98]	valid's binary_logloss: 0.109248
    [99]	valid's binary_logloss: 0.10897
    [100]	valid's binary_logloss: 0.108694
    Did not meet early stopping. Best iteration is:
    [100]	valid's binary_logloss: 0.108694
    [1]	valid's binary_logloss: 0.110714
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.0971756
    [3]	valid's binary_logloss: 0.0875393
    [4]	valid's binary_logloss: 0.0799311
    [5]	valid's binary_logloss: 0.0736387
    [6]	valid's binary_logloss: 0.0682669
    [7]	valid's binary_logloss: 0.063558
    [8]	valid's binary_logloss: 0.0593954
    [9]	valid's binary_logloss: 0.0556757
    [10]	valid's binary_logloss: 0.0522967
    [11]	valid's binary_logloss: 0.0492235
    [12]	valid's binary_logloss: 0.0464131
    [13]	valid's binary_logloss: 0.0438313
    [14]	valid's binary_logloss: 0.0414637
    [15]	valid's binary_logloss: 0.0392666
    [16]	valid's binary_logloss: 0.037222
    [17]	valid's binary_logloss: 0.0353342
    [18]	valid's binary_logloss: 0.0335621
    [19]	valid's binary_logloss: 0.0319086
    [20]	valid's binary_logloss: 0.0303766
    [21]	valid's binary_logloss: 0.0289294
    [22]	valid's binary_logloss: 0.0275736
    [23]	valid's binary_logloss: 0.0263022
    [24]	valid's binary_logloss: 0.0251216
    [25]	valid's binary_logloss: 0.024011
    [26]	valid's binary_logloss: 0.0229542
    [27]	valid's binary_logloss: 0.021968
    [28]	valid's binary_logloss: 0.0210457
    [29]	valid's binary_logloss: 0.0201632
    [30]	valid's binary_logloss: 0.0193421
    [31]	valid's binary_logloss: 0.0185652
    [32]	valid's binary_logloss: 0.017838
    [33]	valid's binary_logloss: 0.0171493
    [34]	valid's binary_logloss: 0.0164881
    [35]	valid's binary_logloss: 0.0158642
    [36]	valid's binary_logloss: 0.015284
    [37]	valid's binary_logloss: 0.0147377
    [38]	valid's binary_logloss: 0.014225
    [39]	valid's binary_logloss: 0.0137269
    [40]	valid's binary_logloss: 0.0132674
    [41]	valid's binary_logloss: 0.0128207
    [42]	valid's binary_logloss: 0.0123987
    [43]	valid's binary_logloss: 0.0119999
    [44]	valid's binary_logloss: 0.011638
    [45]	valid's binary_logloss: 0.011281
    [46]	valid's binary_logloss: 0.0109553
    [47]	valid's binary_logloss: 0.0106356
    [48]	valid's binary_logloss: 0.0103334
    [49]	valid's binary_logloss: 0.0100479
    [50]	valid's binary_logloss: 0.00979199
    [51]	valid's binary_logloss: 0.00954598
    [52]	valid's binary_logloss: 0.00931506
    [53]	valid's binary_logloss: 0.00908569
    [54]	valid's binary_logloss: 0.00888394
    [55]	valid's binary_logloss: 0.00869023
    [56]	valid's binary_logloss: 0.00849558
    [57]	valid's binary_logloss: 0.0083118
    [58]	valid's binary_logloss: 0.00813831
    [59]	valid's binary_logloss: 0.00798633
    [60]	valid's binary_logloss: 0.00783111
    [61]	valid's binary_logloss: 0.00768467
    [62]	valid's binary_logloss: 0.00755551
    [63]	valid's binary_logloss: 0.00743787
    [64]	valid's binary_logloss: 0.00732349
    [65]	valid's binary_logloss: 0.00721711
    [66]	valid's binary_logloss: 0.00711991
    [67]	valid's binary_logloss: 0.0070274
    [68]	valid's binary_logloss: 0.00692774
    [69]	valid's binary_logloss: 0.00683562
    [70]	valid's binary_logloss: 0.00674905
    [71]	valid's binary_logloss: 0.00666772
    [72]	valid's binary_logloss: 0.00659136
    [73]	valid's binary_logloss: 0.00651789
    [74]	valid's binary_logloss: 0.00645671
    [75]	valid's binary_logloss: 0.00639147
    [76]	valid's binary_logloss: 0.00633024
    [77]	valid's binary_logloss: 0.00627281
    [78]	valid's binary_logloss: 0.00621898
    [79]	valid's binary_logloss: 0.00616854
    [80]	valid's binary_logloss: 0.00612132
    [81]	valid's binary_logloss: 0.00607714
    [82]	valid's binary_logloss: 0.00603583
    [83]	valid's binary_logloss: 0.00599723
    [84]	valid's binary_logloss: 0.00596118
    [85]	valid's binary_logloss: 0.00593265
    [86]	valid's binary_logloss: 0.00590088
    [87]	valid's binary_logloss: 0.00587745
    [88]	valid's binary_logloss: 0.00584951
    [89]	valid's binary_logloss: 0.00582352
    [90]	valid's binary_logloss: 0.00579937
    [91]	valid's binary_logloss: 0.00578211
    [92]	valid's binary_logloss: 0.00576097
    [93]	valid's binary_logloss: 0.00574648
    [94]	valid's binary_logloss: 0.005728
    [95]	valid's binary_logloss: 0.0057156
    [96]	valid's binary_logloss: 0.00569951
    [97]	valid's binary_logloss: 0.00568465
    [98]	valid's binary_logloss: 0.00567095
    [99]	valid's binary_logloss: 0.00566395
    [100]	valid's binary_logloss: 0.00565209
    [101]	valid's binary_logloss: 0.0056412
    [102]	valid's binary_logloss: 0.00563122
    [103]	valid's binary_logloss: 0.00562536
    [104]	valid's binary_logloss: 0.00561679
    [105]	valid's binary_logloss: 0.00560897
    [106]	valid's binary_logloss: 0.00560184
    [107]	valid's binary_logloss: 0.00559703
    [108]	valid's binary_logloss: 0.00559098
    [109]	valid's binary_logloss: 0.00558781
    [110]	valid's binary_logloss: 0.0055827
    [111]	valid's binary_logloss: 0.0055781
    [112]	valid's binary_logloss: 0.00557396
    [113]	valid's binary_logloss: 0.00557026
    [114]	valid's binary_logloss: 0.00556696
    [115]	valid's binary_logloss: 0.00556403
    [116]	valid's binary_logloss: 0.00556144
    [117]	valid's binary_logloss: 0.00555916
    [118]	valid's binary_logloss: 0.00555717
    [119]	valid's binary_logloss: 0.00555737
    [120]	valid's binary_logloss: 0.00555685
    [121]	valid's binary_logloss: 0.00555541
    [122]	valid's binary_logloss: 0.0055542
    [123]	valid's binary_logloss: 0.00555318
    [124]	valid's binary_logloss: 0.00555235
    [125]	valid's binary_logloss: 0.00555167
    [126]	valid's binary_logloss: 0.00555115
    [127]	valid's binary_logloss: 0.00555076
    [128]	valid's binary_logloss: 0.00555049
    [129]	valid's binary_logloss: 0.00555033
    [130]	valid's binary_logloss: 0.00555027
    [131]	valid's binary_logloss: 0.00555007
    [132]	valid's binary_logloss: 0.00554995
    [133]	valid's binary_logloss: 0.00555007
    [134]	valid's binary_logloss: 0.00555025
    [135]	valid's binary_logloss: 0.00555049
    [136]	valid's binary_logloss: 0.00555078
    [137]	valid's binary_logloss: 0.00555112
    [138]	valid's binary_logloss: 0.0055515
    [139]	valid's binary_logloss: 0.00555192
    [140]	valid's binary_logloss: 0.00555236
    [141]	valid's binary_logloss: 0.00555282
    [142]	valid's binary_logloss: 0.00555296
    Early stopping, best iteration is:
    [132]	valid's binary_logloss: 0.00554995
    [1]	valid's binary_logloss: 0.110593
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.0970863
    [3]	valid's binary_logloss: 0.0874429
    [4]	valid's binary_logloss: 0.0798485
    [5]	valid's binary_logloss: 0.0735602
    [6]	valid's binary_logloss: 0.0681962
    [7]	valid's binary_logloss: 0.063493
    [8]	valid's binary_logloss: 0.0593326
    [9]	valid's binary_logloss: 0.0556003
    [10]	valid's binary_logloss: 0.0522284
    [11]	valid's binary_logloss: 0.0491665
    [12]	valid's binary_logloss: 0.0463734
    [13]	valid's binary_logloss: 0.0437928
    [14]	valid's binary_logloss: 0.0414133
    [15]	valid's binary_logloss: 0.0392199
    [16]	valid's binary_logloss: 0.0371901
    [17]	valid's binary_logloss: 0.0353044
    [18]	valid's binary_logloss: 0.0335331
    [19]	valid's binary_logloss: 0.0318805
    [20]	valid's binary_logloss: 0.0303362
    [21]	valid's binary_logloss: 0.0288978
    [22]	valid's binary_logloss: 0.0275428
    [23]	valid's binary_logloss: 0.0262721
    [24]	valid's binary_logloss: 0.0250925
    [25]	valid's binary_logloss: 0.0239808
    [26]	valid's binary_logloss: 0.022925
    [27]	valid's binary_logloss: 0.02194
    [28]	valid's binary_logloss: 0.0210143
    [29]	valid's binary_logloss: 0.0201411
    [30]	valid's binary_logloss: 0.0193102
    [31]	valid's binary_logloss: 0.018534
    [32]	valid's binary_logloss: 0.0177943
    [33]	valid's binary_logloss: 0.0171053
    [34]	valid's binary_logloss: 0.0164461
    [35]	valid's binary_logloss: 0.015824
    [36]	valid's binary_logloss: 0.0152437
    [37]	valid's binary_logloss: 0.0146883
    [38]	valid's binary_logloss: 0.0141638
    [39]	valid's binary_logloss: 0.0136685
    [40]	valid's binary_logloss: 0.0132077
    [41]	valid's binary_logloss: 0.0127648
    [42]	valid's binary_logloss: 0.0123465
    [43]	valid's binary_logloss: 0.0119644
    [44]	valid's binary_logloss: 0.0115902
    [45]	valid's binary_logloss: 0.0112368
    [46]	valid's binary_logloss: 0.0109096
    [47]	valid's binary_logloss: 0.0105933
    [48]	valid's binary_logloss: 0.0102945
    [49]	valid's binary_logloss: 0.0100205
    [50]	valid's binary_logloss: 0.00975323
    [51]	valid's binary_logloss: 0.0095077
    [52]	valid's binary_logloss: 0.00926861
    [53]	valid's binary_logloss: 0.00904289
    [54]	valid's binary_logloss: 0.00883894
    [55]	valid's binary_logloss: 0.00864415
    [56]	valid's binary_logloss: 0.00846167
    [57]	valid's binary_logloss: 0.00828098
    [58]	valid's binary_logloss: 0.00811055
    [59]	valid's binary_logloss: 0.00795688
    [60]	valid's binary_logloss: 0.0078047
    [61]	valid's binary_logloss: 0.00766127
    [62]	valid's binary_logloss: 0.00753491
    [63]	valid's binary_logloss: 0.00741523
    [64]	valid's binary_logloss: 0.0073011
    [65]	valid's binary_logloss: 0.00719549
    [66]	valid's binary_logloss: 0.00709941
    [67]	valid's binary_logloss: 0.00700403
    [68]	valid's binary_logloss: 0.00690744
    [69]	valid's binary_logloss: 0.00681666
    [70]	valid's binary_logloss: 0.00673139
    [71]	valid's binary_logloss: 0.00665133
    [72]	valid's binary_logloss: 0.00658749
    [73]	valid's binary_logloss: 0.00652273
    [74]	valid's binary_logloss: 0.00646311
    [75]	valid's binary_logloss: 0.0064076
    [76]	valid's binary_logloss: 0.00634798
    [77]	valid's binary_logloss: 0.00629221
    [78]	valid's binary_logloss: 0.00624006
    [79]	valid's binary_logloss: 0.00619135
    [80]	valid's binary_logloss: 0.00614589
    [81]	valid's binary_logloss: 0.00610349
    [82]	valid's binary_logloss: 0.00606399
    [83]	valid's binary_logloss: 0.00602722
    [84]	valid's binary_logloss: 0.00599304
    [85]	valid's binary_logloss: 0.00596656
    [86]	valid's binary_logloss: 0.00593672
    [87]	valid's binary_logloss: 0.0059128
    [88]	valid's binary_logloss: 0.00588685
    [89]	valid's binary_logloss: 0.00586285
    [90]	valid's binary_logloss: 0.0058407
    [91]	valid's binary_logloss: 0.0058244
    [92]	valid's binary_logloss: 0.0058053
    [93]	valid's binary_logloss: 0.00579066
    [94]	valid's binary_logloss: 0.00577426
    [95]	valid's binary_logloss: 0.00576533
    [96]	valid's binary_logloss: 0.00575132
    [97]	valid's binary_logloss: 0.00573854
    [98]	valid's binary_logloss: 0.00572692
    [99]	valid's binary_logloss: 0.00571851
    [100]	valid's binary_logloss: 0.00570873
    [101]	valid's binary_logloss: 0.0056999
    [102]	valid's binary_logloss: 0.00569197
    [103]	valid's binary_logloss: 0.00568716
    [104]	valid's binary_logloss: 0.00568062
    [105]	valid's binary_logloss: 0.00567482
    [106]	valid's binary_logloss: 0.00566969
    [107]	valid's binary_logloss: 0.00566657
    [108]	valid's binary_logloss: 0.00566249
    [109]	valid's binary_logloss: 0.00565983
    [110]	valid's binary_logloss: 0.00565666
    [111]	valid's binary_logloss: 0.00565398
    [112]	valid's binary_logloss: 0.00565174
    [113]	valid's binary_logloss: 0.00564992
    [114]	valid's binary_logloss: 0.00564847
    [115]	valid's binary_logloss: 0.00564736
    [116]	valid's binary_logloss: 0.00564657
    [117]	valid's binary_logloss: 0.00564607
    [118]	valid's binary_logloss: 0.00564584
    [119]	valid's binary_logloss: 0.00564731
    [120]	valid's binary_logloss: 0.00564772
    [121]	valid's binary_logloss: 0.00564797
    [122]	valid's binary_logloss: 0.00564842
    [123]	valid's binary_logloss: 0.00564903
    [124]	valid's binary_logloss: 0.0056498
    [125]	valid's binary_logloss: 0.0056507
    [126]	valid's binary_logloss: 0.00565171
    [127]	valid's binary_logloss: 0.00565284
    [128]	valid's binary_logloss: 0.00565405
    Early stopping, best iteration is:
    [118]	valid's binary_logloss: 0.00564584
    [1]	valid's binary_logloss: 0.110687
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.0971621
    [3]	valid's binary_logloss: 0.0875064
    [4]	valid's binary_logloss: 0.0799053
    [5]	valid's binary_logloss: 0.0736167
    [6]	valid's binary_logloss: 0.0682448
    [7]	valid's binary_logloss: 0.0635388
    [8]	valid's binary_logloss: 0.0593782
    [9]	valid's binary_logloss: 0.0556442
    [10]	valid's binary_logloss: 0.052271
    [11]	valid's binary_logloss: 0.0492031
    [12]	valid's binary_logloss: 0.0464034
    [13]	valid's binary_logloss: 0.0438246
    [14]	valid's binary_logloss: 0.0414543
    [15]	valid's binary_logloss: 0.0392613
    [16]	valid's binary_logloss: 0.0372313
    [17]	valid's binary_logloss: 0.035341
    [18]	valid's binary_logloss: 0.033569
    [19]	valid's binary_logloss: 0.0319159
    [20]	valid's binary_logloss: 0.0303712
    [21]	valid's binary_logloss: 0.028926
    [22]	valid's binary_logloss: 0.0275808
    [23]	valid's binary_logloss: 0.0263097
    [24]	valid's binary_logloss: 0.0251292
    [25]	valid's binary_logloss: 0.0240213
    [26]	valid's binary_logloss: 0.0229651
    [27]	valid's binary_logloss: 0.0219807
    [28]	valid's binary_logloss: 0.0210558
    [29]	valid's binary_logloss: 0.0201831
    [30]	valid's binary_logloss: 0.0193615
    [31]	valid's binary_logloss: 0.0185869
    [32]	valid's binary_logloss: 0.0178585
    [33]	valid's binary_logloss: 0.0171683
    [34]	valid's binary_logloss: 0.0165073
    [35]	valid's binary_logloss: 0.0158827
    [36]	valid's binary_logloss: 0.0153046
    [37]	valid's binary_logloss: 0.0147468
    [38]	valid's binary_logloss: 0.0142301
    [39]	valid's binary_logloss: 0.0137314
    [40]	valid's binary_logloss: 0.0132604
    [41]	valid's binary_logloss: 0.0128153
    [42]	valid's binary_logloss: 0.0123948
    [43]	valid's binary_logloss: 0.0120072
    [44]	valid's binary_logloss: 0.0116308
    [45]	valid's binary_logloss: 0.011275
    [46]	valid's binary_logloss: 0.01095
    [47]	valid's binary_logloss: 0.0106315
    [48]	valid's binary_logloss: 0.0103405
    [49]	valid's binary_logloss: 0.0100552
    [50]	valid's binary_logloss: 0.00978557
    [51]	valid's binary_logloss: 0.00954229
    [52]	valid's binary_logloss: 0.00931247
    [53]	valid's binary_logloss: 0.00909376
    [54]	valid's binary_logloss: 0.00888868
    [55]	valid's binary_logloss: 0.00869364
    [56]	valid's binary_logloss: 0.00850944
    [57]	valid's binary_logloss: 0.00833735
    [58]	valid's binary_logloss: 0.00816266
    [59]	valid's binary_logloss: 0.00800856
    [60]	valid's binary_logloss: 0.00785223
    [61]	valid's binary_logloss: 0.00770474
    [62]	valid's binary_logloss: 0.0075766
    [63]	valid's binary_logloss: 0.00745686
    [64]	valid's binary_logloss: 0.00734293
    [65]	valid's binary_logloss: 0.00723781
    [66]	valid's binary_logloss: 0.00713653
    [67]	valid's binary_logloss: 0.00703997
    [68]	valid's binary_logloss: 0.00695134
    [69]	valid's binary_logloss: 0.00685604
    [70]	valid's binary_logloss: 0.00676636
    [71]	valid's binary_logloss: 0.00668199
    [72]	valid's binary_logloss: 0.00660265
    [73]	valid's binary_logloss: 0.00652808
    [74]	valid's binary_logloss: 0.00646745
    [75]	valid's binary_logloss: 0.00640113
    [76]	valid's binary_logloss: 0.00633887
    [77]	valid's binary_logloss: 0.00628046
    [78]	valid's binary_logloss: 0.00622568
    [79]	valid's binary_logloss: 0.00617434
    [80]	valid's binary_logloss: 0.00612625
    [81]	valid's binary_logloss: 0.00608123
    [82]	valid's binary_logloss: 0.00603911
    [83]	valid's binary_logloss: 0.00599974
    [84]	valid's binary_logloss: 0.00596295
    [85]	valid's binary_logloss: 0.00593737
    [86]	valid's binary_logloss: 0.0059049
    [87]	valid's binary_logloss: 0.00588066
    [88]	valid's binary_logloss: 0.00585207
    [89]	valid's binary_logloss: 0.00582545
    [90]	valid's binary_logloss: 0.00580069
    [91]	valid's binary_logloss: 0.00578245
    [92]	valid's binary_logloss: 0.00576074
    [93]	valid's binary_logloss: 0.00574572
    [94]	valid's binary_logloss: 0.00572672
    [95]	valid's binary_logloss: 0.00571346
    [96]	valid's binary_logloss: 0.00569687
    [97]	valid's binary_logloss: 0.00568153
    [98]	valid's binary_logloss: 0.00566738
    [99]	valid's binary_logloss: 0.00565767
    [100]	valid's binary_logloss: 0.00564537
    [101]	valid's binary_logloss: 0.00563407
    [102]	valid's binary_logloss: 0.00562368
    [103]	valid's binary_logloss: 0.00561886
    [104]	valid's binary_logloss: 0.00560991
    [105]	valid's binary_logloss: 0.00560173
    [106]	valid's binary_logloss: 0.00559426
    [107]	valid's binary_logloss: 0.00559011
    [108]	valid's binary_logloss: 0.00558374
    [109]	valid's binary_logloss: 0.00558012
    [110]	valid's binary_logloss: 0.0055747
    [111]	valid's binary_logloss: 0.00556981
    [112]	valid's binary_logloss: 0.0055654
    [113]	valid's binary_logloss: 0.00556144
    [114]	valid's binary_logloss: 0.0055579
    [115]	valid's binary_logloss: 0.00555474
    [116]	valid's binary_logloss: 0.00555194
    [117]	valid's binary_logloss: 0.00554946
    [118]	valid's binary_logloss: 0.00554729
    [119]	valid's binary_logloss: 0.00554616
    [120]	valid's binary_logloss: 0.0055452
    [121]	valid's binary_logloss: 0.00554359
    [122]	valid's binary_logloss: 0.00554221
    [123]	valid's binary_logloss: 0.00554104
    [124]	valid's binary_logloss: 0.00554007
    [125]	valid's binary_logloss: 0.00553927
    [126]	valid's binary_logloss: 0.00553863
    [127]	valid's binary_logloss: 0.00553814
    [128]	valid's binary_logloss: 0.00553777
    [129]	valid's binary_logloss: 0.00553753
    [130]	valid's binary_logloss: 0.00553739
    [131]	valid's binary_logloss: 0.0055376
    [132]	valid's binary_logloss: 0.00553767
    [133]	valid's binary_logloss: 0.00553771
    [134]	valid's binary_logloss: 0.00553782
    [135]	valid's binary_logloss: 0.005538
    [136]	valid's binary_logloss: 0.00553824
    [137]	valid's binary_logloss: 0.00553854
    [138]	valid's binary_logloss: 0.00553888
    [139]	valid's binary_logloss: 0.00553926
    [140]	valid's binary_logloss: 0.00553967
    Early stopping, best iteration is:
    [130]	valid's binary_logloss: 0.00553739
    [1]	valid's binary_logloss: 0.239791
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.239791
    [3]	valid's binary_logloss: 0.239791
    [4]	valid's binary_logloss: 0.239791
    [5]	valid's binary_logloss: 0.239791
    [6]	valid's binary_logloss: 0.239791
    [7]	valid's binary_logloss: 0.239791
    [8]	valid's binary_logloss: 0.239791
    [9]	valid's binary_logloss: 0.239791
    [10]	valid's binary_logloss: 0.239791
    [11]	valid's binary_logloss: 0.239791
    Early stopping, best iteration is:
    [1]	valid's binary_logloss: 0.239791
    [1]	valid's binary_logloss: 0.239791
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.239791
    [3]	valid's binary_logloss: 0.239791
    [4]	valid's binary_logloss: 0.239791
    [5]	valid's binary_logloss: 0.239791
    [6]	valid's binary_logloss: 0.239791
    [7]	valid's binary_logloss: 0.239791
    [8]	valid's binary_logloss: 0.239791
    [9]	valid's binary_logloss: 0.239791
    [10]	valid's binary_logloss: 0.239791
    [11]	valid's binary_logloss: 0.239791
    Early stopping, best iteration is:
    [1]	valid's binary_logloss: 0.239791
    [1]	valid's binary_logloss: 0.239792
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.239792
    [3]	valid's binary_logloss: 0.239792
    [4]	valid's binary_logloss: 0.239792
    [5]	valid's binary_logloss: 0.239792
    [6]	valid's binary_logloss: 0.239792
    [7]	valid's binary_logloss: 0.239792
    [8]	valid's binary_logloss: 0.239792
    [9]	valid's binary_logloss: 0.239792
    [10]	valid's binary_logloss: 0.239792
    [11]	valid's binary_logloss: 0.239792
    Early stopping, best iteration is:
    [1]	valid's binary_logloss: 0.239792
    [1]	valid's binary_logloss: 0.0719235
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.0628067
    [3]	valid's binary_logloss: 0.0554644
    [4]	valid's binary_logloss: 0.0493447
    [5]	valid's binary_logloss: 0.0441669
    [6]	valid's binary_logloss: 0.0397133
    [7]	valid's binary_logloss: 0.0358351
    [8]	valid's binary_logloss: 0.0324466
    [9]	valid's binary_logloss: 0.029469
    [10]	valid's binary_logloss: 0.0268361
    [11]	valid's binary_logloss: 0.0245067
    [12]	valid's binary_logloss: 0.0224453
    [13]	valid's binary_logloss: 0.0206081
    [14]	valid's binary_logloss: 0.0189764
    [15]	valid's binary_logloss: 0.0175353
    [16]	valid's binary_logloss: 0.0162298
    [17]	valid's binary_logloss: 0.0150818
    [18]	valid's binary_logloss: 0.0140425
    [19]	valid's binary_logloss: 0.0131088
    [20]	valid's binary_logloss: 0.0122743
    [21]	valid's binary_logloss: 0.0115259
    [22]	valid's binary_logloss: 0.0108724
    [23]	valid's binary_logloss: 0.0102715
    [24]	valid's binary_logloss: 0.00974595
    [25]	valid's binary_logloss: 0.00927776
    [26]	valid's binary_logloss: 0.00884497
    [27]	valid's binary_logloss: 0.00846665
    [28]	valid's binary_logloss: 0.00813231
    [29]	valid's binary_logloss: 0.00782347
    [30]	valid's binary_logloss: 0.00756583
    [31]	valid's binary_logloss: 0.00734306
    [32]	valid's binary_logloss: 0.00713262
    [33]	valid's binary_logloss: 0.00695719
    [34]	valid's binary_logloss: 0.00678033
    [35]	valid's binary_logloss: 0.00662384
    [36]	valid's binary_logloss: 0.00648998
    [37]	valid's binary_logloss: 0.00636748
    [38]	valid's binary_logloss: 0.00625962
    [39]	valid's binary_logloss: 0.00618366
    [40]	valid's binary_logloss: 0.00609886
    [41]	valid's binary_logloss: 0.00603464
    [42]	valid's binary_logloss: 0.0059685
    [43]	valid's binary_logloss: 0.00591087
    [44]	valid's binary_logloss: 0.00586076
    [45]	valid's binary_logloss: 0.0058173
    [46]	valid's binary_logloss: 0.00579255
    [47]	valid's binary_logloss: 0.00575892
    [48]	valid's binary_logloss: 0.00572995
    [49]	valid's binary_logloss: 0.00570749
    [50]	valid's binary_logloss: 0.00568583
    [51]	valid's binary_logloss: 0.00567538
    [52]	valid's binary_logloss: 0.00565877
    [53]	valid's binary_logloss: 0.00564899
    [54]	valid's binary_logloss: 0.00564436
    [55]	valid's binary_logloss: 0.00563305
    [56]	valid's binary_logloss: 0.00562347
    [57]	valid's binary_logloss: 0.00561528
    [58]	valid's binary_logloss: 0.00560841
    [59]	valid's binary_logloss: 0.00560518
    [60]	valid's binary_logloss: 0.00559982
    [61]	valid's binary_logloss: 0.00559535
    [62]	valid's binary_logloss: 0.00559377
    [63]	valid's binary_logloss: 0.00559151
    [64]	valid's binary_logloss: 0.00558996
    [65]	valid's binary_logloss: 0.00558868
    [66]	valid's binary_logloss: 0.00558809
    [67]	valid's binary_logloss: 0.00558568
    [68]	valid's binary_logloss: 0.0055837
    [69]	valid's binary_logloss: 0.00558288
    [70]	valid's binary_logloss: 0.00558127
    [71]	valid's binary_logloss: 0.00558119
    [72]	valid's binary_logloss: 0.0055799
    [73]	valid's binary_logloss: 0.00557945
    [74]	valid's binary_logloss: 0.00557907
    [75]	valid's binary_logloss: 0.00557875
    [76]	valid's binary_logloss: 0.00557848
    [77]	valid's binary_logloss: 0.00557828
    [78]	valid's binary_logloss: 0.00557807
    [79]	valid's binary_logloss: 0.00557794
    [80]	valid's binary_logloss: 0.00557716
    [81]	valid's binary_logloss: 0.00557734
    [82]	valid's binary_logloss: 0.00557666
    [83]	valid's binary_logloss: 0.00557656
    [84]	valid's binary_logloss: 0.005576
    [85]	valid's binary_logloss: 0.00557592
    [86]	valid's binary_logloss: 0.00557585
    [87]	valid's binary_logloss: 0.00557579
    [88]	valid's binary_logloss: 0.00557574
    [89]	valid's binary_logloss: 0.00557529
    [90]	valid's binary_logloss: 0.0055749
    [91]	valid's binary_logloss: 0.00557487
    [92]	valid's binary_logloss: 0.00557447
    [93]	valid's binary_logloss: 0.00557447
    [94]	valid's binary_logloss: 0.00557418
    [95]	valid's binary_logloss: 0.00557418
    [96]	valid's binary_logloss: 0.00557387
    [97]	valid's binary_logloss: 0.00557387
    [98]	valid's binary_logloss: 0.00557363
    [99]	valid's binary_logloss: 0.00557363
    [100]	valid's binary_logloss: 0.00557363
    Did not meet early stopping. Best iteration is:
    [98]	valid's binary_logloss: 0.00557363
    [1]	valid's binary_logloss: 0.071758
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.0626625
    [3]	valid's binary_logloss: 0.0553314
    [4]	valid's binary_logloss: 0.0492206
    [5]	valid's binary_logloss: 0.0440554
    [6]	valid's binary_logloss: 0.0396041
    [7]	valid's binary_logloss: 0.0357287
    [8]	valid's binary_logloss: 0.0323426
    [9]	valid's binary_logloss: 0.0293678
    [10]	valid's binary_logloss: 0.0267444
    [11]	valid's binary_logloss: 0.024415
    [12]	valid's binary_logloss: 0.0223566
    [13]	valid's binary_logloss: 0.0205196
    [14]	valid's binary_logloss: 0.0188922
    [15]	valid's binary_logloss: 0.0174488
    [16]	valid's binary_logloss: 0.016146
    [17]	valid's binary_logloss: 0.0150007
    [18]	valid's binary_logloss: 0.0139695
    [19]	valid's binary_logloss: 0.01304
    [20]	valid's binary_logloss: 0.0122067
    [21]	valid's binary_logloss: 0.011462
    [22]	valid's binary_logloss: 0.0108076
    [23]	valid's binary_logloss: 0.0102108
    [24]	valid's binary_logloss: 0.00969124
    [25]	valid's binary_logloss: 0.00922973
    [26]	valid's binary_logloss: 0.00880117
    [27]	valid's binary_logloss: 0.00842628
    [28]	valid's binary_logloss: 0.00809356
    [29]	valid's binary_logloss: 0.00778913
    [30]	valid's binary_logloss: 0.0075322
    [31]	valid's binary_logloss: 0.00731199
    [32]	valid's binary_logloss: 0.00710412
    [33]	valid's binary_logloss: 0.00693218
    [34]	valid's binary_logloss: 0.00675979
    [35]	valid's binary_logloss: 0.00660785
    [36]	valid's binary_logloss: 0.00647562
    [37]	valid's binary_logloss: 0.00635759
    [38]	valid's binary_logloss: 0.00625422
    [39]	valid's binary_logloss: 0.00617651
    [40]	valid's binary_logloss: 0.00609652
    [41]	valid's binary_logloss: 0.00603479
    [42]	valid's binary_logloss: 0.00597324
    [43]	valid's binary_logloss: 0.00592011
    [44]	valid's binary_logloss: 0.00587442
    [45]	valid's binary_logloss: 0.00583527
    [46]	valid's binary_logloss: 0.00581222
    [47]	valid's binary_logloss: 0.00578277
    [48]	valid's binary_logloss: 0.00575784
    [49]	valid's binary_logloss: 0.00573976
    [50]	valid's binary_logloss: 0.00572167
    [51]	valid's binary_logloss: 0.00571404
    [52]	valid's binary_logloss: 0.00570079
    [53]	valid's binary_logloss: 0.00569425
    [54]	valid's binary_logloss: 0.00569025
    [55]	valid's binary_logloss: 0.00568208
    [56]	valid's binary_logloss: 0.00567547
    [57]	valid's binary_logloss: 0.00567019
    [58]	valid's binary_logloss: 0.00566602
    [59]	valid's binary_logloss: 0.00566488
    [60]	valid's binary_logloss: 0.00566203
    [61]	valid's binary_logloss: 0.00565989
    [62]	valid's binary_logloss: 0.00566007
    [63]	valid's binary_logloss: 0.00565952
    [64]	valid's binary_logloss: 0.0056592
    [65]	valid's binary_logloss: 0.00565895
    [66]	valid's binary_logloss: 0.0056597
    [67]	valid's binary_logloss: 0.00565901
    [68]	valid's binary_logloss: 0.00565858
    [69]	valid's binary_logloss: 0.00565936
    [70]	valid's binary_logloss: 0.00565917
    [71]	valid's binary_logloss: 0.00565958
    [72]	valid's binary_logloss: 0.00565953
    [73]	valid's binary_logloss: 0.00565983
    [74]	valid's binary_logloss: 0.00566012
    [75]	valid's binary_logloss: 0.00566041
    [76]	valid's binary_logloss: 0.00566087
    [77]	valid's binary_logloss: 0.00566134
    [78]	valid's binary_logloss: 0.0056616
    Early stopping, best iteration is:
    [68]	valid's binary_logloss: 0.00565858
    [1]	valid's binary_logloss: 0.0718028
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.0626998
    [3]	valid's binary_logloss: 0.0553657
    [4]	valid's binary_logloss: 0.0492565
    [5]	valid's binary_logloss: 0.044116
    [6]	valid's binary_logloss: 0.0396644
    [7]	valid's binary_logloss: 0.0357874
    [8]	valid's binary_logloss: 0.0324
    [9]	valid's binary_logloss: 0.0294248
    [10]	valid's binary_logloss: 0.0267923
    [11]	valid's binary_logloss: 0.0244633
    [12]	valid's binary_logloss: 0.0224041
    [13]	valid's binary_logloss: 0.0205674
    [14]	valid's binary_logloss: 0.0189394
    [15]	valid's binary_logloss: 0.0174989
    [16]	valid's binary_logloss: 0.0161995
    [17]	valid's binary_logloss: 0.015053
    [18]	valid's binary_logloss: 0.0140209
    [19]	valid's binary_logloss: 0.0130898
    [20]	valid's binary_logloss: 0.0122574
    [21]	valid's binary_logloss: 0.0115132
    [22]	valid's binary_logloss: 0.0108581
    [23]	valid's binary_logloss: 0.010257
    [24]	valid's binary_logloss: 0.0097351
    [25]	valid's binary_logloss: 0.00926846
    [26]	valid's binary_logloss: 0.0088355
    [27]	valid's binary_logloss: 0.0084567
    [28]	valid's binary_logloss: 0.00812143
    [29]	valid's binary_logloss: 0.00781274
    [30]	valid's binary_logloss: 0.00755824
    [31]	valid's binary_logloss: 0.00733686
    [32]	valid's binary_logloss: 0.00712496
    [33]	valid's binary_logloss: 0.00695195
    [34]	valid's binary_logloss: 0.00677466
    [35]	valid's binary_logloss: 0.00661785
    [36]	valid's binary_logloss: 0.00648351
    [37]	valid's binary_logloss: 0.00636077
    [38]	valid's binary_logloss: 0.00625272
    [39]	valid's binary_logloss: 0.00617256
    [40]	valid's binary_logloss: 0.00608791
    [41]	valid's binary_logloss: 0.00602353
    [42]	valid's binary_logloss: 0.0059575
    [43]	valid's binary_logloss: 0.00589995
    [44]	valid's binary_logloss: 0.00584991
    [45]	valid's binary_logloss: 0.00580648
    [46]	valid's binary_logloss: 0.00577923
    [47]	valid's binary_logloss: 0.00574585
    [48]	valid's binary_logloss: 0.00571709
    [49]	valid's binary_logloss: 0.00569607
    [50]	valid's binary_logloss: 0.00567452
    [51]	valid's binary_logloss: 0.00566303
    [52]	valid's binary_logloss: 0.00564671
    [53]	valid's binary_logloss: 0.00563569
    [54]	valid's binary_logloss: 0.00563017
    [55]	valid's binary_logloss: 0.00561908
    [56]	valid's binary_logloss: 0.00560968
    [57]	valid's binary_logloss: 0.00560173
    [58]	valid's binary_logloss: 0.00559502
    [59]	valid's binary_logloss: 0.0055921
    [60]	valid's binary_logloss: 0.00558693
    [61]	valid's binary_logloss: 0.00558258
    [62]	valid's binary_logloss: 0.00558052
    [63]	valid's binary_logloss: 0.00557885
    [64]	valid's binary_logloss: 0.00557749
    [65]	valid's binary_logloss: 0.00557618
    [66]	valid's binary_logloss: 0.00557528
    [67]	valid's binary_logloss: 0.00557292
    [68]	valid's binary_logloss: 0.00557093
    [69]	valid's binary_logloss: 0.00557046
    [70]	valid's binary_logloss: 0.00556887
    [71]	valid's binary_logloss: 0.00556857
    [72]	valid's binary_logloss: 0.00556727
    [73]	valid's binary_logloss: 0.00556707
    [74]	valid's binary_logloss: 0.00556692
    [75]	valid's binary_logloss: 0.0055668
    [76]	valid's binary_logloss: 0.0055662
    [77]	valid's binary_logloss: 0.00556571
    [78]	valid's binary_logloss: 0.00556566
    [79]	valid's binary_logloss: 0.00556527
    [80]	valid's binary_logloss: 0.00556495
    [81]	valid's binary_logloss: 0.00556504
    [82]	valid's binary_logloss: 0.00556479
    [83]	valid's binary_logloss: 0.00556478
    [84]	valid's binary_logloss: 0.00556416
    [85]	valid's binary_logloss: 0.00556416
    [86]	valid's binary_logloss: 0.00556416
    [87]	valid's binary_logloss: 0.00556416
    [88]	valid's binary_logloss: 0.00556417
    [89]	valid's binary_logloss: 0.00556362
    [90]	valid's binary_logloss: 0.00556313
    [91]	valid's binary_logloss: 0.00556314
    [92]	valid's binary_logloss: 0.00556271
    [93]	valid's binary_logloss: 0.00556271
    [94]	valid's binary_logloss: 0.00556234
    [95]	valid's binary_logloss: 0.00556234
    [96]	valid's binary_logloss: 0.00556201
    [97]	valid's binary_logloss: 0.00556201
    [98]	valid's binary_logloss: 0.00556173
    [99]	valid's binary_logloss: 0.00556173
    [100]	valid's binary_logloss: 0.00556173
    Did not meet early stopping. Best iteration is:
    [98]	valid's binary_logloss: 0.00556173
    [1]	valid's binary_logloss: 0.239791
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.239791
    [3]	valid's binary_logloss: 0.239791
    [4]	valid's binary_logloss: 0.239791
    [5]	valid's binary_logloss: 0.239791
    [6]	valid's binary_logloss: 0.239791
    [7]	valid's binary_logloss: 0.239791
    [8]	valid's binary_logloss: 0.239791
    [9]	valid's binary_logloss: 0.239791
    [10]	valid's binary_logloss: 0.239791
    [11]	valid's binary_logloss: 0.239791
    Early stopping, best iteration is:
    [1]	valid's binary_logloss: 0.239791
    [1]	valid's binary_logloss: 0.239791
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.239791
    [3]	valid's binary_logloss: 0.239791
    [4]	valid's binary_logloss: 0.239791
    [5]	valid's binary_logloss: 0.239791
    [6]	valid's binary_logloss: 0.239791
    [7]	valid's binary_logloss: 0.239791
    [8]	valid's binary_logloss: 0.239791
    [9]	valid's binary_logloss: 0.239791
    [10]	valid's binary_logloss: 0.239791
    [11]	valid's binary_logloss: 0.239791
    Early stopping, best iteration is:
    [1]	valid's binary_logloss: 0.239791
    [1]	valid's binary_logloss: 0.239792
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.239792
    [3]	valid's binary_logloss: 0.239792
    [4]	valid's binary_logloss: 0.239792
    [5]	valid's binary_logloss: 0.239792
    [6]	valid's binary_logloss: 0.239792
    [7]	valid's binary_logloss: 0.239792
    [8]	valid's binary_logloss: 0.239792
    [9]	valid's binary_logloss: 0.239792
    [10]	valid's binary_logloss: 0.239792
    [11]	valid's binary_logloss: 0.239792
    Early stopping, best iteration is:
    [1]	valid's binary_logloss: 0.239792
    [1]	valid's binary_logloss: 0.100287
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.082402
    [3]	valid's binary_logloss: 0.0704493
    [4]	valid's binary_logloss: 0.0614543
    [5]	valid's binary_logloss: 0.0542825
    [6]	valid's binary_logloss: 0.0483828
    [7]	valid's binary_logloss: 0.0433916
    [8]	valid's binary_logloss: 0.0391298
    [9]	valid's binary_logloss: 0.0354533
    [10]	valid's binary_logloss: 0.0322514
    [11]	valid's binary_logloss: 0.0294503
    [12]	valid's binary_logloss: 0.0269775
    [13]	valid's binary_logloss: 0.0247807
    [14]	valid's binary_logloss: 0.0228329
    [15]	valid's binary_logloss: 0.0211041
    [16]	valid's binary_logloss: 0.0195671
    [17]	valid's binary_logloss: 0.0182067
    [18]	valid's binary_logloss: 0.0169814
    [19]	valid's binary_logloss: 0.0158747
    [20]	valid's binary_logloss: 0.0148835
    [21]	valid's binary_logloss: 0.0139949
    [22]	valid's binary_logloss: 0.0131977
    [23]	valid's binary_logloss: 0.0124818
    [24]	valid's binary_logloss: 0.0118473
    [25]	valid's binary_logloss: 0.0112843
    [26]	valid's binary_logloss: 0.0107609
    [27]	valid's binary_logloss: 0.0102915
    [28]	valid's binary_logloss: 0.00987093
    [29]	valid's binary_logloss: 0.00949425
    [30]	valid's binary_logloss: 0.00914658
    [31]	valid's binary_logloss: 0.00884195
    [32]	valid's binary_logloss: 0.0085618
    [33]	valid's binary_logloss: 0.00831878
    [34]	valid's binary_logloss: 0.0080825
    [35]	valid's binary_logloss: 0.00786859
    [36]	valid's binary_logloss: 0.00768152
    [37]	valid's binary_logloss: 0.00751404
    [38]	valid's binary_logloss: 0.00735303
    [39]	valid's binary_logloss: 0.00720684
    [40]	valid's binary_logloss: 0.00707397
    [41]	valid's binary_logloss: 0.00695312
    [42]	valid's binary_logloss: 0.0068431
    [43]	valid's binary_logloss: 0.00674285
    [44]	valid's binary_logloss: 0.00665142
    [45]	valid's binary_logloss: 0.00656796
    [46]	valid's binary_logloss: 0.0064978
    [47]	valid's binary_logloss: 0.00642759
    [48]	valid's binary_logloss: 0.00636332
    [49]	valid's binary_logloss: 0.00630443
    [50]	valid's binary_logloss: 0.00625043
    [51]	valid's binary_logloss: 0.00620331
    [52]	valid's binary_logloss: 0.00616379
    [53]	valid's binary_logloss: 0.00612136
    [54]	valid's binary_logloss: 0.00608596
    [55]	valid's binary_logloss: 0.00604973
    [56]	valid's binary_logloss: 0.00601632
    [57]	valid's binary_logloss: 0.00598549
    [58]	valid's binary_logloss: 0.00595701
    [59]	valid's binary_logloss: 0.00593212
    [60]	valid's binary_logloss: 0.00590757
    [61]	valid's binary_logloss: 0.00588484
    [62]	valid's binary_logloss: 0.00586487
    [63]	valid's binary_logloss: 0.00584757
    [64]	valid's binary_logloss: 0.00583041
    [65]	valid's binary_logloss: 0.00581555
    [66]	valid's binary_logloss: 0.00580345
    [67]	valid's binary_logloss: 0.00578833
    [68]	valid's binary_logloss: 0.00577425
    [69]	valid's binary_logloss: 0.00576114
    [70]	valid's binary_logloss: 0.00574891
    [71]	valid's binary_logloss: 0.00573749
    [72]	valid's binary_logloss: 0.00572684
    [73]	valid's binary_logloss: 0.00571837
    [74]	valid's binary_logloss: 0.0057098
    [75]	valid's binary_logloss: 0.00570248
    [76]	valid's binary_logloss: 0.00569413
    [77]	valid's binary_logloss: 0.00568632
    [78]	valid's binary_logloss: 0.0056803
    [79]	valid's binary_logloss: 0.00567339
    [80]	valid's binary_logloss: 0.00566689
    [81]	valid's binary_logloss: 0.0056608
    [82]	valid's binary_logloss: 0.00565507
    [83]	valid's binary_logloss: 0.00564969
    [84]	valid's binary_logloss: 0.00564463
    [85]	valid's binary_logloss: 0.00564191
    [86]	valid's binary_logloss: 0.00563738
    [87]	valid's binary_logloss: 0.00563408
    [88]	valid's binary_logloss: 0.00563004
    [89]	valid's binary_logloss: 0.00562623
    [90]	valid's binary_logloss: 0.00562263
    [91]	valid's binary_logloss: 0.00562006
    [92]	valid's binary_logloss: 0.00561683
    [93]	valid's binary_logloss: 0.0056146
    [94]	valid's binary_logloss: 0.00561169
    [95]	valid's binary_logloss: 0.00560969
    [96]	valid's binary_logloss: 0.00560707
    [97]	valid's binary_logloss: 0.00560459
    [98]	valid's binary_logloss: 0.00560224
    [99]	valid's binary_logloss: 0.0056007
    [100]	valid's binary_logloss: 0.00559858
    [101]	valid's binary_logloss: 0.00559657
    [102]	valid's binary_logloss: 0.00559467
    [103]	valid's binary_logloss: 0.00559403
    [104]	valid's binary_logloss: 0.0055923
    [105]	valid's binary_logloss: 0.00559066
    [106]	valid's binary_logloss: 0.00558911
    [107]	valid's binary_logloss: 0.0055878
    [108]	valid's binary_logloss: 0.00558658
    [109]	valid's binary_logloss: 0.00558572
    [110]	valid's binary_logloss: 0.0055845
    [111]	valid's binary_logloss: 0.00558334
    [112]	valid's binary_logloss: 0.00558262
    [113]	valid's binary_logloss: 0.00558158
    [114]	valid's binary_logloss: 0.0055806
    [115]	valid's binary_logloss: 0.00557967
    [116]	valid's binary_logloss: 0.00557869
    [117]	valid's binary_logloss: 0.00557776
    [118]	valid's binary_logloss: 0.00557687
    [119]	valid's binary_logloss: 0.00557676
    [120]	valid's binary_logloss: 0.00557631
    [121]	valid's binary_logloss: 0.00557561
    [122]	valid's binary_logloss: 0.00557486
    [123]	valid's binary_logloss: 0.00557423
    [124]	valid's binary_logloss: 0.00557355
    [125]	valid's binary_logloss: 0.0055729
    [126]	valid's binary_logloss: 0.00557228
    [127]	valid's binary_logloss: 0.0055717
    [128]	valid's binary_logloss: 0.00557113
    [129]	valid's binary_logloss: 0.0055706
    [130]	valid's binary_logloss: 0.00557015
    [131]	valid's binary_logloss: 0.0055697
    [132]	valid's binary_logloss: 0.00556946
    [133]	valid's binary_logloss: 0.00556906
    [134]	valid's binary_logloss: 0.00556868
    [135]	valid's binary_logloss: 0.00556826
    [136]	valid's binary_logloss: 0.00556786
    [137]	valid's binary_logloss: 0.00556748
    [138]	valid's binary_logloss: 0.00556711
    [139]	valid's binary_logloss: 0.00556676
    [140]	valid's binary_logloss: 0.00556643
    [141]	valid's binary_logloss: 0.00556616
    [142]	valid's binary_logloss: 0.00556583
    [143]	valid's binary_logloss: 0.00556557
    [144]	valid's binary_logloss: 0.00556527
    [145]	valid's binary_logloss: 0.00556504
    [146]	valid's binary_logloss: 0.00556481
    [147]	valid's binary_logloss: 0.0055646
    [148]	valid's binary_logloss: 0.0055644
    [149]	valid's binary_logloss: 0.00556414
    [150]	valid's binary_logloss: 0.00556395
    [151]	valid's binary_logloss: 0.00556377
    [152]	valid's binary_logloss: 0.0055636
    [153]	valid's binary_logloss: 0.00556344
    [154]	valid's binary_logloss: 0.00556325
    [155]	valid's binary_logloss: 0.00556304
    [156]	valid's binary_logloss: 0.00556289
    [157]	valid's binary_logloss: 0.00556275
    [158]	valid's binary_logloss: 0.00556262
    [159]	valid's binary_logloss: 0.00556243
    [160]	valid's binary_logloss: 0.00556226
    [161]	valid's binary_logloss: 0.00556214
    [162]	valid's binary_logloss: 0.00556203
    [163]	valid's binary_logloss: 0.00556192
    [164]	valid's binary_logloss: 0.00556182
    [165]	valid's binary_logloss: 0.00556189
    [166]	valid's binary_logloss: 0.00556179
    [167]	valid's binary_logloss: 0.0055617
    [168]	valid's binary_logloss: 0.00556156
    [169]	valid's binary_logloss: 0.00556147
    [170]	valid's binary_logloss: 0.00556134
    [171]	valid's binary_logloss: 0.00556126
    [172]	valid's binary_logloss: 0.00556119
    [173]	valid's binary_logloss: 0.00556125
    [174]	valid's binary_logloss: 0.00556114
    [175]	valid's binary_logloss: 0.00556107
    [176]	valid's binary_logloss: 0.00556101
    [177]	valid's binary_logloss: 0.0055609
    [178]	valid's binary_logloss: 0.00556084
    [179]	valid's binary_logloss: 0.00556079
    [180]	valid's binary_logloss: 0.00556073
    [181]	valid's binary_logloss: 0.00556078
    [182]	valid's binary_logloss: 0.00556073
    [183]	valid's binary_logloss: 0.00556064
    [184]	valid's binary_logloss: 0.0055606
    [185]	valid's binary_logloss: 0.00556055
    [186]	valid's binary_logloss: 0.0055606
    [187]	valid's binary_logloss: 0.00556056
    [188]	valid's binary_logloss: 0.00556052
    [189]	valid's binary_logloss: 0.00556048
    [190]	valid's binary_logloss: 0.00556044
    [191]	valid's binary_logloss: 0.00556037
    [192]	valid's binary_logloss: 0.00556034
    [193]	valid's binary_logloss: 0.00556031
    [194]	valid's binary_logloss: 0.00556025
    [195]	valid's binary_logloss: 0.00556022
    [196]	valid's binary_logloss: 0.00556019
    [197]	valid's binary_logloss: 0.00556013
    [198]	valid's binary_logloss: 0.00556011
    [199]	valid's binary_logloss: 0.00556008
    [200]	valid's binary_logloss: 0.00556005
    [201]	valid's binary_logloss: 0.00556003
    [202]	valid's binary_logloss: 0.00555998
    [203]	valid's binary_logloss: 0.00555993
    [204]	valid's binary_logloss: 0.00555997
    [205]	valid's binary_logloss: 0.00555995
    [206]	valid's binary_logloss: 0.00555993
    [207]	valid's binary_logloss: 0.00555991
    [208]	valid's binary_logloss: 0.0055599
    [209]	valid's binary_logloss: 0.00555986
    [210]	valid's binary_logloss: 0.00555984
    [211]	valid's binary_logloss: 0.00555983
    [212]	valid's binary_logloss: 0.00555981
    [213]	valid's binary_logloss: 0.0055598
    [214]	valid's binary_logloss: 0.00555979
    [215]	valid's binary_logloss: 0.00555978
    [216]	valid's binary_logloss: 0.00555976
    [217]	valid's binary_logloss: 0.00555975
    [218]	valid's binary_logloss: 0.00555974
    [219]	valid's binary_logloss: 0.00555973
    [220]	valid's binary_logloss: 0.00555972
    [221]	valid's binary_logloss: 0.00555971
    [222]	valid's binary_logloss: 0.0055597
    [223]	valid's binary_logloss: 0.00555969
    [224]	valid's binary_logloss: 0.00555968
    [225]	valid's binary_logloss: 0.00555968
    [226]	valid's binary_logloss: 0.00555967
    [227]	valid's binary_logloss: 0.00555966
    [228]	valid's binary_logloss: 0.00555965
    [229]	valid's binary_logloss: 0.00555965
    [230]	valid's binary_logloss: 0.00555962
    [231]	valid's binary_logloss: 0.00555962
    [232]	valid's binary_logloss: 0.00555961
    [233]	valid's binary_logloss: 0.00555961
    [234]	valid's binary_logloss: 0.00555958
    [235]	valid's binary_logloss: 0.00555958
    [236]	valid's binary_logloss: 0.00555958
    [237]	valid's binary_logloss: 0.00555958
    [238]	valid's binary_logloss: 0.00555957
    [239]	valid's binary_logloss: 0.00555957
    [240]	valid's binary_logloss: 0.00555955
    [241]	valid's binary_logloss: 0.00555955
    [242]	valid's binary_logloss: 0.00555955
    [243]	valid's binary_logloss: 0.00555954
    [244]	valid's binary_logloss: 0.00555954
    [245]	valid's binary_logloss: 0.00555953
    [246]	valid's binary_logloss: 0.00555952
    [247]	valid's binary_logloss: 0.00555951
    [248]	valid's binary_logloss: 0.00555951
    [249]	valid's binary_logloss: 0.00555951
    [250]	valid's binary_logloss: 0.00555951
    [251]	valid's binary_logloss: 0.00555949
    [252]	valid's binary_logloss: 0.00555949
    [253]	valid's binary_logloss: 0.00555949
    [254]	valid's binary_logloss: 0.00555949
    [255]	valid's binary_logloss: 0.00555949
    [256]	valid's binary_logloss: 0.0055595
    [257]	valid's binary_logloss: 0.0055595
    [258]	valid's binary_logloss: 0.00555948
    [259]	valid's binary_logloss: 0.00555948
    [260]	valid's binary_logloss: 0.00555948
    [261]	valid's binary_logloss: 0.00555949
    [262]	valid's binary_logloss: 0.00555947
    [263]	valid's binary_logloss: 0.00555946
    [264]	valid's binary_logloss: 0.00555946
    [265]	valid's binary_logloss: 0.00555947
    [266]	valid's binary_logloss: 0.00555947
    [267]	valid's binary_logloss: 0.00555947
    [268]	valid's binary_logloss: 0.00555947
    [269]	valid's binary_logloss: 0.00555947
    [270]	valid's binary_logloss: 0.00555947
    [271]	valid's binary_logloss: 0.00555947
    [272]	valid's binary_logloss: 0.00555947
    [273]	valid's binary_logloss: 0.00555947
    Early stopping, best iteration is:
    [263]	valid's binary_logloss: 0.00555946
    [1]	valid's binary_logloss: 0.100173
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.082271
    [3]	valid's binary_logloss: 0.0703615
    [4]	valid's binary_logloss: 0.0613897
    [5]	valid's binary_logloss: 0.0542384
    [6]	valid's binary_logloss: 0.0483424
    [7]	valid's binary_logloss: 0.0433645
    [8]	valid's binary_logloss: 0.0391046
    [9]	valid's binary_logloss: 0.0354224
    [10]	valid's binary_logloss: 0.0322162
    [11]	valid's binary_logloss: 0.029419
    [12]	valid's binary_logloss: 0.0269371
    [13]	valid's binary_logloss: 0.0247432
    [14]	valid's binary_logloss: 0.0227981
    [15]	valid's binary_logloss: 0.0210749
    [16]	valid's binary_logloss: 0.0195371
    [17]	valid's binary_logloss: 0.0181699
    [18]	valid's binary_logloss: 0.0169368
    [19]	valid's binary_logloss: 0.0158337
    [20]	valid's binary_logloss: 0.0148457
    [21]	valid's binary_logloss: 0.0139599
    [22]	valid's binary_logloss: 0.0131651
    [23]	valid's binary_logloss: 0.0124514
    [24]	valid's binary_logloss: 0.011814
    [25]	valid's binary_logloss: 0.0112522
    [26]	valid's binary_logloss: 0.0107333
    [27]	valid's binary_logloss: 0.0102691
    [28]	valid's binary_logloss: 0.00985369
    [29]	valid's binary_logloss: 0.00947302
    [30]	valid's binary_logloss: 0.00913157
    [31]	valid's binary_logloss: 0.00882509
    [32]	valid's binary_logloss: 0.00854829
    [33]	valid's binary_logloss: 0.00829923
    [34]	valid's binary_logloss: 0.00806525
    [35]	valid's binary_logloss: 0.00785366
    [36]	valid's binary_logloss: 0.00766981
    [37]	valid's binary_logloss: 0.00749938
    [38]	valid's binary_logloss: 0.00734598
    [39]	valid's binary_logloss: 0.00720155
    [40]	valid's binary_logloss: 0.0070705
    [41]	valid's binary_logloss: 0.00695148
    [42]	valid's binary_logloss: 0.00684332
    [43]	valid's binary_logloss: 0.00674493
    [44]	valid's binary_logloss: 0.00665537
    [45]	valid's binary_logloss: 0.00657377
    [46]	valid's binary_logloss: 0.00650272
    [47]	valid's binary_logloss: 0.00643439
    [48]	valid's binary_logloss: 0.00637197
    [49]	valid's binary_logloss: 0.00631492
    [50]	valid's binary_logloss: 0.00626272
    [51]	valid's binary_logloss: 0.00621761
    [52]	valid's binary_logloss: 0.00617359
    [53]	valid's binary_logloss: 0.00613322
    [54]	valid's binary_logloss: 0.0060989
    [55]	valid's binary_logloss: 0.00606464
    [56]	valid's binary_logloss: 0.00603313
    [57]	valid's binary_logloss: 0.00600415
    [58]	valid's binary_logloss: 0.00597746
    [59]	valid's binary_logloss: 0.00595421
    [60]	valid's binary_logloss: 0.00593137
    [61]	valid's binary_logloss: 0.00591029
    [62]	valid's binary_logloss: 0.00589237
    [63]	valid's binary_logloss: 0.00587609
    [64]	valid's binary_logloss: 0.00586132
    [65]	valid's binary_logloss: 0.00584732
    [66]	valid's binary_logloss: 0.00583485
    [67]	valid's binary_logloss: 0.00582235
    [68]	valid's binary_logloss: 0.00580966
    [69]	valid's binary_logloss: 0.0057979
    [70]	valid's binary_logloss: 0.00578698
    [71]	valid's binary_logloss: 0.00577685
    [72]	valid's binary_logloss: 0.00576744
    [73]	valid's binary_logloss: 0.00575916
    [74]	valid's binary_logloss: 0.00575155
    [75]	valid's binary_logloss: 0.00574667
    [76]	valid's binary_logloss: 0.00573947
    [77]	valid's binary_logloss: 0.00573278
    [78]	valid's binary_logloss: 0.00572777
    [79]	valid's binary_logloss: 0.00572193
    [80]	valid's binary_logloss: 0.00571648
    [81]	valid's binary_logloss: 0.00571141
    [82]	valid's binary_logloss: 0.00570668
    [83]	valid's binary_logloss: 0.00570227
    [84]	valid's binary_logloss: 0.00569816
    [85]	valid's binary_logloss: 0.00569512
    [86]	valid's binary_logloss: 0.00569151
    [87]	valid's binary_logloss: 0.00568907
    [88]	valid's binary_logloss: 0.0056859
    [89]	valid's binary_logloss: 0.00568294
    [90]	valid's binary_logloss: 0.00568019
    [91]	valid's binary_logloss: 0.00567782
    [92]	valid's binary_logloss: 0.00567539
    [93]	valid's binary_logloss: 0.00567336
    [94]	valid's binary_logloss: 0.00567122
    [95]	valid's binary_logloss: 0.00567072
    [96]	valid's binary_logloss: 0.00566885
    [97]	valid's binary_logloss: 0.0056671
    [98]	valid's binary_logloss: 0.00566547
    [99]	valid's binary_logloss: 0.00566447
    [100]	valid's binary_logloss: 0.00566285
    [101]	valid's binary_logloss: 0.00566151
    [102]	valid's binary_logloss: 0.00566026
    [103]	valid's binary_logloss: 0.00565926
    [104]	valid's binary_logloss: 0.00565817
    [105]	valid's binary_logloss: 0.00565715
    [106]	valid's binary_logloss: 0.00565605
    [107]	valid's binary_logloss: 0.0056556
    [108]	valid's binary_logloss: 0.00565521
    [109]	valid's binary_logloss: 0.0056548
    [110]	valid's binary_logloss: 0.00565388
    [111]	valid's binary_logloss: 0.00565303
    [112]	valid's binary_logloss: 0.00565279
    [113]	valid's binary_logloss: 0.00565203
    [114]	valid's binary_logloss: 0.00565133
    [115]	valid's binary_logloss: 0.00565069
    [116]	valid's binary_logloss: 0.00565009
    [117]	valid's binary_logloss: 0.00564955
    [118]	valid's binary_logloss: 0.00564915
    [119]	valid's binary_logloss: 0.00564907
    [120]	valid's binary_logloss: 0.00564881
    [121]	valid's binary_logloss: 0.0056484
    [122]	valid's binary_logloss: 0.00564811
    [123]	valid's binary_logloss: 0.00564785
    [124]	valid's binary_logloss: 0.00564761
    [125]	valid's binary_logloss: 0.00564731
    [126]	valid's binary_logloss: 0.00564712
    [127]	valid's binary_logloss: 0.00564695
    [128]	valid's binary_logloss: 0.00564679
    [129]	valid's binary_logloss: 0.00564666
    [130]	valid's binary_logloss: 0.00564654
    [131]	valid's binary_logloss: 0.00564662
    [132]	valid's binary_logloss: 0.00564672
    [133]	valid's binary_logloss: 0.00564657
    [134]	valid's binary_logloss: 0.00564644
    [135]	valid's binary_logloss: 0.00564639
    [136]	valid's binary_logloss: 0.00564635
    [137]	valid's binary_logloss: 0.00564632
    [138]	valid's binary_logloss: 0.00564624
    [139]	valid's binary_logloss: 0.00564623
    [140]	valid's binary_logloss: 0.00564624
    [141]	valid's binary_logloss: 0.00564625
    [142]	valid's binary_logloss: 0.00564639
    [143]	valid's binary_logloss: 0.00564636
    [144]	valid's binary_logloss: 0.00564652
    [145]	valid's binary_logloss: 0.0056465
    [146]	valid's binary_logloss: 0.00564649
    [147]	valid's binary_logloss: 0.00564649
    [148]	valid's binary_logloss: 0.0056465
    [149]	valid's binary_logloss: 0.00564667
    Early stopping, best iteration is:
    [139]	valid's binary_logloss: 0.00564623
    [1]	valid's binary_logloss: 0.100276
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.0823561
    [3]	valid's binary_logloss: 0.0704358
    [4]	valid's binary_logloss: 0.0614603
    [5]	valid's binary_logloss: 0.0543071
    [6]	valid's binary_logloss: 0.0483926
    [7]	valid's binary_logloss: 0.0434155
    [8]	valid's binary_logloss: 0.0391528
    [9]	valid's binary_logloss: 0.0354723
    [10]	valid's binary_logloss: 0.032264
    [11]	valid's binary_logloss: 0.0294599
    [12]	valid's binary_logloss: 0.0269766
    [13]	valid's binary_logloss: 0.0247816
    [14]	valid's binary_logloss: 0.0228355
    [15]	valid's binary_logloss: 0.0211123
    [16]	valid's binary_logloss: 0.0195763
    [17]	valid's binary_logloss: 0.0182077
    [18]	valid's binary_logloss: 0.0169737
    [19]	valid's binary_logloss: 0.0158697
    [20]	valid's binary_logloss: 0.0148809
    [21]	valid's binary_logloss: 0.0139943
    [22]	valid's binary_logloss: 0.0131988
    [23]	valid's binary_logloss: 0.0124844
    [24]	valid's binary_logloss: 0.0118482
    [25]	valid's binary_logloss: 0.0112771
    [26]	valid's binary_logloss: 0.0107546
    [27]	valid's binary_logloss: 0.0102876
    [28]	valid's binary_logloss: 0.00986975
    [29]	valid's binary_logloss: 0.00949129
    [30]	valid's binary_logloss: 0.00914209
    [31]	valid's binary_logloss: 0.00883612
    [32]	valid's binary_logloss: 0.0085586
    [33]	valid's binary_logloss: 0.00830989
    [34]	valid's binary_logloss: 0.00807401
    [35]	valid's binary_logloss: 0.00786705
    [36]	valid's binary_logloss: 0.00768078
    [37]	valid's binary_logloss: 0.00751108
    [38]	valid's binary_logloss: 0.00734956
    [39]	valid's binary_logloss: 0.00720288
    [40]	valid's binary_logloss: 0.00706955
    [41]	valid's binary_logloss: 0.00694826
    [42]	valid's binary_logloss: 0.00683781
    [43]	valid's binary_logloss: 0.00673716
    [44]	valid's binary_logloss: 0.00664534
    [45]	valid's binary_logloss: 0.00656151
    [46]	valid's binary_logloss: 0.00649287
    [47]	valid's binary_logloss: 0.00642233
    [48]	valid's binary_logloss: 0.00635775
    [49]	valid's binary_logloss: 0.00629856
    [50]	valid's binary_logloss: 0.00624426
    [51]	valid's binary_logloss: 0.00619779
    [52]	valid's binary_logloss: 0.00615172
    [53]	valid's binary_logloss: 0.00610933
    [54]	valid's binary_logloss: 0.00607337
    [55]	valid's binary_logloss: 0.00603716
    [56]	valid's binary_logloss: 0.00600375
    [57]	valid's binary_logloss: 0.00597291
    [58]	valid's binary_logloss: 0.0059444
    [59]	valid's binary_logloss: 0.00591971
    [60]	valid's binary_logloss: 0.00589513
    [61]	valid's binary_logloss: 0.00587236
    [62]	valid's binary_logloss: 0.00585328
    [63]	valid's binary_logloss: 0.00583613
    [64]	valid's binary_logloss: 0.00581994
    [65]	valid's binary_logloss: 0.00580409
    [66]	valid's binary_logloss: 0.0057897
    [67]	valid's binary_logloss: 0.00577704
    [68]	valid's binary_logloss: 0.00576284
    [69]	valid's binary_logloss: 0.00574961
    [70]	valid's binary_logloss: 0.00573727
    [71]	valid's binary_logloss: 0.00572574
    [72]	valid's binary_logloss: 0.00571497
    [73]	valid's binary_logloss: 0.00570658
    [74]	valid's binary_logloss: 0.00569845
    [75]	valid's binary_logloss: 0.00569045
    [76]	valid's binary_logloss: 0.00568202
    [77]	valid's binary_logloss: 0.00567412
    [78]	valid's binary_logloss: 0.00566766
    [79]	valid's binary_logloss: 0.00566065
    [80]	valid's binary_logloss: 0.00565407
    [81]	valid's binary_logloss: 0.00564789
    [82]	valid's binary_logloss: 0.00564208
    [83]	valid's binary_logloss: 0.00563662
    [84]	valid's binary_logloss: 0.00563147
    [85]	valid's binary_logloss: 0.00562845
    [86]	valid's binary_logloss: 0.00562386
    [87]	valid's binary_logloss: 0.00562038
    [88]	valid's binary_logloss: 0.00561627
    [89]	valid's binary_logloss: 0.00561239
    [90]	valid's binary_logloss: 0.00560873
    [91]	valid's binary_logloss: 0.0056062
    [92]	valid's binary_logloss: 0.00560291
    [93]	valid's binary_logloss: 0.00560083
    [94]	valid's binary_logloss: 0.00559787
    [95]	valid's binary_logloss: 0.00559576
    [96]	valid's binary_logloss: 0.00559309
    [97]	valid's binary_logloss: 0.00559057
    [98]	valid's binary_logloss: 0.00558817
    [99]	valid's binary_logloss: 0.00558636
    [100]	valid's binary_logloss: 0.00558419
    [101]	valid's binary_logloss: 0.00558214
    [102]	valid's binary_logloss: 0.00558019
    [103]	valid's binary_logloss: 0.00557946
    [104]	valid's binary_logloss: 0.00557769
    [105]	valid's binary_logloss: 0.00557601
    [106]	valid's binary_logloss: 0.00557442
    [107]	valid's binary_logloss: 0.00557341
    [108]	valid's binary_logloss: 0.00557267
    [109]	valid's binary_logloss: 0.00557164
    [110]	valid's binary_logloss: 0.00557031
    [111]	valid's binary_logloss: 0.00556904
    [112]	valid's binary_logloss: 0.00556838
    [113]	valid's binary_logloss: 0.00556708
    [114]	valid's binary_logloss: 0.00556598
    [115]	valid's binary_logloss: 0.00556481
    [116]	valid's binary_logloss: 0.00556381
    [117]	valid's binary_logloss: 0.00556286
    [118]	valid's binary_logloss: 0.00556195
    [119]	valid's binary_logloss: 0.00556183
    [120]	valid's binary_logloss: 0.00556126
    [121]	valid's binary_logloss: 0.00556034
    [122]	valid's binary_logloss: 0.00555947
    [123]	valid's binary_logloss: 0.00555874
    [124]	valid's binary_logloss: 0.00555805
    [125]	valid's binary_logloss: 0.00555729
    [126]	valid's binary_logloss: 0.00555666
    [127]	valid's binary_logloss: 0.00555605
    [128]	valid's binary_logloss: 0.00555548
    [129]	valid's binary_logloss: 0.00555493
    [130]	valid's binary_logloss: 0.0055544
    [131]	valid's binary_logloss: 0.0055541
    [132]	valid's binary_logloss: 0.00555388
    [133]	valid's binary_logloss: 0.00555333
    [134]	valid's binary_logloss: 0.00555281
    [135]	valid's binary_logloss: 0.00555231
    [136]	valid's binary_logloss: 0.0055519
    [137]	valid's binary_logloss: 0.00555151
    [138]	valid's binary_logloss: 0.00555114
    [139]	valid's binary_logloss: 0.00555073
    [140]	valid's binary_logloss: 0.00555038
    [141]	valid's binary_logloss: 0.00555006
    [142]	valid's binary_logloss: 0.00554997
    [143]	valid's binary_logloss: 0.00554961
    [144]	valid's binary_logloss: 0.00554955
    [145]	valid's binary_logloss: 0.00554921
    [146]	valid's binary_logloss: 0.00554888
    [147]	valid's binary_logloss: 0.00554858
    [148]	valid's binary_logloss: 0.00554828
    [149]	valid's binary_logloss: 0.00554825
    [150]	valid's binary_logloss: 0.00554797
    [151]	valid's binary_logloss: 0.00554771
    [152]	valid's binary_logloss: 0.00554746
    [153]	valid's binary_logloss: 0.00554722
    [154]	valid's binary_logloss: 0.00554699
    [155]	valid's binary_logloss: 0.00554698
    [156]	valid's binary_logloss: 0.00554676
    [157]	valid's binary_logloss: 0.00554656
    [158]	valid's binary_logloss: 0.00554636
    [159]	valid's binary_logloss: 0.00554636
    [160]	valid's binary_logloss: 0.00554637
    [161]	valid's binary_logloss: 0.00554618
    [162]	valid's binary_logloss: 0.00554601
    [163]	valid's binary_logloss: 0.00554584
    [164]	valid's binary_logloss: 0.00554568
    [165]	valid's binary_logloss: 0.00554569
    [166]	valid's binary_logloss: 0.00554554
    [167]	valid's binary_logloss: 0.0055454
    [168]	valid's binary_logloss: 0.00554542
    [169]	valid's binary_logloss: 0.00554528
    [170]	valid's binary_logloss: 0.00554531
    [171]	valid's binary_logloss: 0.00554518
    [172]	valid's binary_logloss: 0.00554505
    [173]	valid's binary_logloss: 0.00554508
    [174]	valid's binary_logloss: 0.00554511
    [175]	valid's binary_logloss: 0.00554499
    [176]	valid's binary_logloss: 0.00554488
    [177]	valid's binary_logloss: 0.00554492
    [178]	valid's binary_logloss: 0.00554481
    [179]	valid's binary_logloss: 0.00554471
    [180]	valid's binary_logloss: 0.00554461
    [181]	valid's binary_logloss: 0.00554464
    [182]	valid's binary_logloss: 0.00554455
    [183]	valid's binary_logloss: 0.00554459
    [184]	valid's binary_logloss: 0.0055445
    [185]	valid's binary_logloss: 0.00554442
    [186]	valid's binary_logloss: 0.00554444
    [187]	valid's binary_logloss: 0.00554436
    [188]	valid's binary_logloss: 0.00554428
    [189]	valid's binary_logloss: 0.00554421
    [190]	valid's binary_logloss: 0.00554414
    [191]	valid's binary_logloss: 0.00554418
    [192]	valid's binary_logloss: 0.00554412
    [193]	valid's binary_logloss: 0.00554406
    [194]	valid's binary_logloss: 0.0055441
    [195]	valid's binary_logloss: 0.00554404
    [196]	valid's binary_logloss: 0.00554398
    [197]	valid's binary_logloss: 0.00554403
    [198]	valid's binary_logloss: 0.00554397
    [199]	valid's binary_logloss: 0.00554399
    [200]	valid's binary_logloss: 0.00554394
    [201]	valid's binary_logloss: 0.00554389
    [202]	valid's binary_logloss: 0.00554393
    [203]	valid's binary_logloss: 0.00554397
    [204]	valid's binary_logloss: 0.00554399
    [205]	valid's binary_logloss: 0.00554394
    [206]	valid's binary_logloss: 0.0055439
    [207]	valid's binary_logloss: 0.00554385
    [208]	valid's binary_logloss: 0.00554381
    [209]	valid's binary_logloss: 0.00554385
    [210]	valid's binary_logloss: 0.00554381
    [211]	valid's binary_logloss: 0.00554377
    [212]	valid's binary_logloss: 0.00554374
    [213]	valid's binary_logloss: 0.0055437
    [214]	valid's binary_logloss: 0.00554367
    [215]	valid's binary_logloss: 0.00554364
    [216]	valid's binary_logloss: 0.00554367
    [217]	valid's binary_logloss: 0.00554364
    [218]	valid's binary_logloss: 0.00554361
    [219]	valid's binary_logloss: 0.00554363
    [220]	valid's binary_logloss: 0.0055436
    [221]	valid's binary_logloss: 0.00554363
    [222]	valid's binary_logloss: 0.0055436
    [223]	valid's binary_logloss: 0.00554358
    [224]	valid's binary_logloss: 0.00554356
    [225]	valid's binary_logloss: 0.00554354
    [226]	valid's binary_logloss: 0.00554352
    [227]	valid's binary_logloss: 0.0055435
    [228]	valid's binary_logloss: 0.00554352
    [229]	valid's binary_logloss: 0.0055435
    [230]	valid's binary_logloss: 0.00554354
    [231]	valid's binary_logloss: 0.00554352
    [232]	valid's binary_logloss: 0.00554351
    [233]	valid's binary_logloss: 0.00554349
    [234]	valid's binary_logloss: 0.00554353
    [235]	valid's binary_logloss: 0.00554351
    [236]	valid's binary_logloss: 0.0055435
    [237]	valid's binary_logloss: 0.00554349
    [238]	valid's binary_logloss: 0.00554348
    [239]	valid's binary_logloss: 0.00554347
    [240]	valid's binary_logloss: 0.0055435
    [241]	valid's binary_logloss: 0.00554349
    [242]	valid's binary_logloss: 0.00554348
    [243]	valid's binary_logloss: 0.00554347
    [244]	valid's binary_logloss: 0.00554346
    [245]	valid's binary_logloss: 0.00554349
    [246]	valid's binary_logloss: 0.00554348
    [247]	valid's binary_logloss: 0.00554351
    [248]	valid's binary_logloss: 0.0055435
    [249]	valid's binary_logloss: 0.00554349
    [250]	valid's binary_logloss: 0.00554348
    [251]	valid's binary_logloss: 0.00554351
    [252]	valid's binary_logloss: 0.0055435
    [253]	valid's binary_logloss: 0.0055435
    [254]	valid's binary_logloss: 0.00554349
    Early stopping, best iteration is:
    [244]	valid's binary_logloss: 0.00554346
    [1]	valid's binary_logloss: 0.111791
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.097951
    [3]	valid's binary_logloss: 0.0881603
    [4]	valid's binary_logloss: 0.0804723
    [5]	valid's binary_logloss: 0.0741081
    [6]	valid's binary_logloss: 0.0686847
    [7]	valid's binary_logloss: 0.0639436
    [8]	valid's binary_logloss: 0.0597465
    [9]	valid's binary_logloss: 0.055989
    [10]	valid's binary_logloss: 0.0525941
    [11]	valid's binary_logloss: 0.0495082
    [12]	valid's binary_logloss: 0.0466945
    [13]	valid's binary_logloss: 0.0440998
    [14]	valid's binary_logloss: 0.0417076
    [15]	valid's binary_logloss: 0.039496
    [16]	valid's binary_logloss: 0.0374439
    [17]	valid's binary_logloss: 0.0355449
    [18]	valid's binary_logloss: 0.0337668
    [19]	valid's binary_logloss: 0.0321077
    [20]	valid's binary_logloss: 0.0305668
    [21]	valid's binary_logloss: 0.0291149
    [22]	valid's binary_logloss: 0.0277548
    [23]	valid's binary_logloss: 0.0264793
    [24]	valid's binary_logloss: 0.0252917
    [25]	valid's binary_logloss: 0.0241709
    [26]	valid's binary_logloss: 0.0231231
    [27]	valid's binary_logloss: 0.0221262
    [28]	valid's binary_logloss: 0.0211879
    [29]	valid's binary_logloss: 0.0203126
    [30]	valid's binary_logloss: 0.0194787
    [31]	valid's binary_logloss: 0.0186926
    [32]	valid's binary_logloss: 0.0179624
    [33]	valid's binary_logloss: 0.0172622
    [34]	valid's binary_logloss: 0.0166099
    [35]	valid's binary_logloss: 0.0159853
    [36]	valid's binary_logloss: 0.0153957
    [37]	valid's binary_logloss: 0.0148389
    [38]	valid's binary_logloss: 0.0143216
    [39]	valid's binary_logloss: 0.0138229
    [40]	valid's binary_logloss: 0.0133516
    [41]	valid's binary_logloss: 0.0129185
    [42]	valid's binary_logloss: 0.0124969
    [43]	valid's binary_logloss: 0.0121082
    [44]	valid's binary_logloss: 0.0117307
    [45]	valid's binary_logloss: 0.011374
    [46]	valid's binary_logloss: 0.0110367
    [47]	valid's binary_logloss: 0.0107294
    [48]	valid's binary_logloss: 0.0104275
    [49]	valid's binary_logloss: 0.0101516
    [50]	valid's binary_logloss: 0.00988106
    [51]	valid's binary_logloss: 0.00962539
    [52]	valid's binary_logloss: 0.00938379
    [53]	valid's binary_logloss: 0.00916763
    [54]	valid's binary_logloss: 0.00896088
    [55]	valid's binary_logloss: 0.00875579
    [56]	valid's binary_logloss: 0.00856206
    [57]	valid's binary_logloss: 0.00837911
    [58]	valid's binary_logloss: 0.00820634
    [59]	valid's binary_logloss: 0.00804324
    [60]	valid's binary_logloss: 0.00789838
    [61]	valid's binary_logloss: 0.00775242
    [62]	valid's binary_logloss: 0.00761469
    [63]	valid's binary_logloss: 0.00749605
    [64]	valid's binary_logloss: 0.00737296
    [65]	valid's binary_logloss: 0.0072569
    [66]	valid's binary_logloss: 0.0071475
    [67]	valid's binary_logloss: 0.00705271
    [68]	valid's binary_logloss: 0.00695501
    [69]	valid's binary_logloss: 0.00686299
    [70]	valid's binary_logloss: 0.00677635
    [71]	valid's binary_logloss: 0.00670391
    [72]	valid's binary_logloss: 0.0066267
    [73]	valid's binary_logloss: 0.00655408
    [74]	valid's binary_logloss: 0.0064858
    [75]	valid's binary_logloss: 0.00642872
    [76]	valid's binary_logloss: 0.00636795
    [77]	valid's binary_logloss: 0.00631087
    [78]	valid's binary_logloss: 0.0062573
    [79]	valid's binary_logloss: 0.00621562
    [80]	valid's binary_logloss: 0.00616807
    [81]	valid's binary_logloss: 0.00612349
    [82]	valid's binary_logloss: 0.00608172
    [83]	valid's binary_logloss: 0.00604259
    [84]	valid's binary_logloss: 0.00601147
    [85]	valid's binary_logloss: 0.00597683
    [86]	valid's binary_logloss: 0.00594444
    [87]	valid's binary_logloss: 0.00591783
    [88]	valid's binary_logloss: 0.0058962
    [89]	valid's binary_logloss: 0.00587409
    [90]	valid's binary_logloss: 0.00584863
    [91]	valid's binary_logloss: 0.00582488
    [92]	valid's binary_logloss: 0.00580274
    [93]	valid's binary_logloss: 0.00578736
    [94]	valid's binary_logloss: 0.00576903
    [95]	valid's binary_logloss: 0.00575089
    [96]	valid's binary_logloss: 0.00573403
    [97]	valid's binary_logloss: 0.00571836
    [98]	valid's binary_logloss: 0.00570382
    [99]	valid's binary_logloss: 0.00569032
    [100]	valid's binary_logloss: 0.00567781
    [101]	valid's binary_logloss: 0.00566622
    [102]	valid's binary_logloss: 0.00565549
    [103]	valid's binary_logloss: 0.00564558
    [104]	valid's binary_logloss: 0.00563641
    [105]	valid's binary_logloss: 0.00562796
    [106]	valid's binary_logloss: 0.00562016
    [107]	valid's binary_logloss: 0.00561298
    [108]	valid's binary_logloss: 0.00560637
    [109]	valid's binary_logloss: 0.0056003
    [110]	valid's binary_logloss: 0.00559472
    [111]	valid's binary_logloss: 0.00558961
    [112]	valid's binary_logloss: 0.00558492
    [113]	valid's binary_logloss: 0.00558064
    [114]	valid's binary_logloss: 0.00557673
    [115]	valid's binary_logloss: 0.00557317
    [116]	valid's binary_logloss: 0.00556993
    [117]	valid's binary_logloss: 0.00556698
    [118]	valid's binary_logloss: 0.0055643
    [119]	valid's binary_logloss: 0.00556189
    [120]	valid's binary_logloss: 0.00556024
    [121]	valid's binary_logloss: 0.00555827
    [122]	valid's binary_logloss: 0.0055565
    [123]	valid's binary_logloss: 0.00555491
    [124]	valid's binary_logloss: 0.00555349
    [125]	valid's binary_logloss: 0.00555223
    [126]	valid's binary_logloss: 0.00555149
    [127]	valid's binary_logloss: 0.00555047
    [128]	valid's binary_logloss: 0.00554957
    [129]	valid's binary_logloss: 0.00554879
    [130]	valid's binary_logloss: 0.00554812
    [131]	valid's binary_logloss: 0.00554837
    [132]	valid's binary_logloss: 0.00554785
    [133]	valid's binary_logloss: 0.00554742
    [134]	valid's binary_logloss: 0.00554739
    [135]	valid's binary_logloss: 0.0055471
    [136]	valid's binary_logloss: 0.00554686
    [137]	valid's binary_logloss: 0.00554678
    [138]	valid's binary_logloss: 0.00554663
    [139]	valid's binary_logloss: 0.00554653
    [140]	valid's binary_logloss: 0.00554648
    [141]	valid's binary_logloss: 0.00554615
    [142]	valid's binary_logloss: 0.00554615
    [143]	valid's binary_logloss: 0.00554619
    [144]	valid's binary_logloss: 0.00554626
    [145]	valid's binary_logloss: 0.00554674
    [146]	valid's binary_logloss: 0.00554685
    [147]	valid's binary_logloss: 0.00554697
    [148]	valid's binary_logloss: 0.00554711
    [149]	valid's binary_logloss: 0.00554726
    [150]	valid's binary_logloss: 0.00554764
    [151]	valid's binary_logloss: 0.00554782
    Early stopping, best iteration is:
    [141]	valid's binary_logloss: 0.00554615
    [1]	valid's binary_logloss: 0.111675
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.0978646
    [3]	valid's binary_logloss: 0.0880812
    [4]	valid's binary_logloss: 0.0804004
    [5]	valid's binary_logloss: 0.0740436
    [6]	valid's binary_logloss: 0.0686251
    [7]	valid's binary_logloss: 0.0638889
    [8]	valid's binary_logloss: 0.0596944
    [9]	valid's binary_logloss: 0.0559392
    [10]	valid's binary_logloss: 0.0525478
    [11]	valid's binary_logloss: 0.0494732
    [12]	valid's binary_logloss: 0.0466515
    [13]	valid's binary_logloss: 0.0440597
    [14]	valid's binary_logloss: 0.0416703
    [15]	valid's binary_logloss: 0.0394603
    [16]	valid's binary_logloss: 0.0374127
    [17]	valid's binary_logloss: 0.035516
    [18]	valid's binary_logloss: 0.0337384
    [19]	valid's binary_logloss: 0.03208
    [20]	valid's binary_logloss: 0.0305303
    [21]	valid's binary_logloss: 0.0290905
    [22]	valid's binary_logloss: 0.0277308
    [23]	valid's binary_logloss: 0.0264559
    [24]	valid's binary_logloss: 0.0252658
    [25]	valid's binary_logloss: 0.0241516
    [26]	valid's binary_logloss: 0.0230989
    [27]	valid's binary_logloss: 0.0221024
    [28]	valid's binary_logloss: 0.0211644
    [29]	valid's binary_logloss: 0.020281
    [30]	valid's binary_logloss: 0.0194546
    [31]	valid's binary_logloss: 0.0186691
    [32]	valid's binary_logloss: 0.0179284
    [33]	valid's binary_logloss: 0.0172297
    [34]	valid's binary_logloss: 0.0165821
    [35]	valid's binary_logloss: 0.0159591
    [36]	valid's binary_logloss: 0.0153763
    [37]	valid's binary_logloss: 0.0148199
    [38]	valid's binary_logloss: 0.0142945
    [39]	valid's binary_logloss: 0.0137982
    [40]	valid's binary_logloss: 0.0133352
    [41]	valid's binary_logloss: 0.0128914
    [42]	valid's binary_logloss: 0.0124721
    [43]	valid's binary_logloss: 0.0120823
    [44]	valid's binary_logloss: 0.0117072
    [45]	valid's binary_logloss: 0.0113529
    [46]	valid's binary_logloss: 0.0110237
    [47]	valid's binary_logloss: 0.0107065
    [48]	valid's binary_logloss: 0.0104069
    [49]	valid's binary_logloss: 0.0101238
    [50]	valid's binary_logloss: 0.0098629
    [51]	valid's binary_logloss: 0.0096096
    [52]	valid's binary_logloss: 0.00937589
    [53]	valid's binary_logloss: 0.00914923
    [54]	valid's binary_logloss: 0.00893521
    [55]	valid's binary_logloss: 0.00873315
    [56]	valid's binary_logloss: 0.00854242
    [57]	valid's binary_logloss: 0.00836786
    [58]	valid's binary_logloss: 0.00819734
    [59]	valid's binary_logloss: 0.00803648
    [60]	valid's binary_logloss: 0.00789535
    [61]	valid's binary_logloss: 0.00775174
    [62]	valid's binary_logloss: 0.00761635
    [63]	valid's binary_logloss: 0.00748876
    [64]	valid's binary_logloss: 0.00737341
    [65]	valid's binary_logloss: 0.00725963
    [66]	valid's binary_logloss: 0.00715251
    [67]	valid's binary_logloss: 0.00705737
    [68]	valid's binary_logloss: 0.00696201
    [69]	valid's binary_logloss: 0.00687233
    [70]	valid's binary_logloss: 0.00678802
    [71]	valid's binary_logloss: 0.00671302
    [72]	valid's binary_logloss: 0.00663812
    [73]	valid's binary_logloss: 0.0065678
    [74]	valid's binary_logloss: 0.0065018
    [75]	valid's binary_logloss: 0.00643991
    [76]	valid's binary_logloss: 0.00638665
    [77]	valid's binary_logloss: 0.00633188
    [78]	valid's binary_logloss: 0.00628411
    [79]	valid's binary_logloss: 0.00623571
    [80]	valid's binary_logloss: 0.00619044
    [81]	valid's binary_logloss: 0.00614813
    [82]	valid's binary_logloss: 0.00610861
    [83]	valid's binary_logloss: 0.00607173
    [84]	valid's binary_logloss: 0.0060402
    [85]	valid's binary_logloss: 0.0060078
    [86]	valid's binary_logloss: 0.00597763
    [87]	valid's binary_logloss: 0.00595575
    [88]	valid's binary_logloss: 0.00592936
    [89]	valid's binary_logloss: 0.00590484
    [90]	valid's binary_logloss: 0.00588209
    [91]	valid's binary_logloss: 0.00586101
    [92]	valid's binary_logloss: 0.00584338
    [93]	valid's binary_logloss: 0.00582506
    [94]	valid's binary_logloss: 0.00580813
    [95]	valid's binary_logloss: 0.00579541
    [96]	valid's binary_logloss: 0.00578079
    [97]	valid's binary_logloss: 0.00576733
    [98]	valid's binary_logloss: 0.00575496
    [99]	valid's binary_logloss: 0.00574361
    [100]	valid's binary_logloss: 0.00573431
    [101]	valid's binary_logloss: 0.00572462
    [102]	valid's binary_logloss: 0.00571576
    [103]	valid's binary_logloss: 0.00570769
    [104]	valid's binary_logloss: 0.00570366
    [105]	valid's binary_logloss: 0.00569686
    [106]	valid's binary_logloss: 0.0056907
    [107]	valid's binary_logloss: 0.00568515
    [108]	valid's binary_logloss: 0.00568015
    [109]	valid's binary_logloss: 0.00567606
    [110]	valid's binary_logloss: 0.00567195
    [111]	valid's binary_logloss: 0.00566829
    [112]	valid's binary_logloss: 0.0056657
    [113]	valid's binary_logloss: 0.00566275
    [114]	valid's binary_logloss: 0.00566017
    [115]	valid's binary_logloss: 0.00565792
    [116]	valid's binary_logloss: 0.00565599
    [117]	valid's binary_logloss: 0.00565434
    [118]	valid's binary_logloss: 0.00565296
    [119]	valid's binary_logloss: 0.00565181
    [120]	valid's binary_logloss: 0.00565073
    [121]	valid's binary_logloss: 0.00564996
    [122]	valid's binary_logloss: 0.00564937
    [123]	valid's binary_logloss: 0.00564895
    [124]	valid's binary_logloss: 0.00564869
    [125]	valid's binary_logloss: 0.00564858
    [126]	valid's binary_logloss: 0.00564855
    [127]	valid's binary_logloss: 0.00564865
    [128]	valid's binary_logloss: 0.00564885
    [129]	valid's binary_logloss: 0.00564914
    [130]	valid's binary_logloss: 0.00564953
    [131]	valid's binary_logloss: 0.00564975
    [132]	valid's binary_logloss: 0.00565026
    [133]	valid's binary_logloss: 0.00565083
    [134]	valid's binary_logloss: 0.00565146
    [135]	valid's binary_logloss: 0.00565277
    [136]	valid's binary_logloss: 0.00565347
    Early stopping, best iteration is:
    [126]	valid's binary_logloss: 0.00564855
    [1]	valid's binary_logloss: 0.111766
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.0979387
    [3]	valid's binary_logloss: 0.0881514
    [4]	valid's binary_logloss: 0.0804629
    [5]	valid's binary_logloss: 0.0741056
    [6]	valid's binary_logloss: 0.0686818
    [7]	valid's binary_logloss: 0.0639473
    [8]	valid's binary_logloss: 0.0597502
    [9]	valid's binary_logloss: 0.0559929
    [10]	valid's binary_logloss: 0.0526
    [11]	valid's binary_logloss: 0.0495177
    [12]	valid's binary_logloss: 0.046696
    [13]	valid's binary_logloss: 0.0441045
    [14]	valid's binary_logloss: 0.0417154
    [15]	valid's binary_logloss: 0.0395098
    [16]	valid's binary_logloss: 0.0374576
    [17]	valid's binary_logloss: 0.03555
    [18]	valid's binary_logloss: 0.0337747
    [19]	valid's binary_logloss: 0.0321231
    [20]	valid's binary_logloss: 0.0305728
    [21]	valid's binary_logloss: 0.0291222
    [22]	valid's binary_logloss: 0.0277726
    [23]	valid's binary_logloss: 0.0264971
    [24]	valid's binary_logloss: 0.025308
    [25]	valid's binary_logloss: 0.0241896
    [26]	valid's binary_logloss: 0.023131
    [27]	valid's binary_logloss: 0.0221451
    [28]	valid's binary_logloss: 0.0212067
    [29]	valid's binary_logloss: 0.0203229
    [30]	valid's binary_logloss: 0.0194982
    [31]	valid's binary_logloss: 0.0187123
    [32]	valid's binary_logloss: 0.0179786
    [33]	valid's binary_logloss: 0.0172784
    [34]	valid's binary_logloss: 0.0166177
    [35]	valid's binary_logloss: 0.0159933
    [36]	valid's binary_logloss: 0.0154123
    [37]	valid's binary_logloss: 0.0148545
    [38]	valid's binary_logloss: 0.0143391
    [39]	valid's binary_logloss: 0.0138406
    [40]	valid's binary_logloss: 0.0133696
    [41]	valid's binary_logloss: 0.0129333
    [42]	valid's binary_logloss: 0.0125117
    [43]	valid's binary_logloss: 0.0121133
    [44]	valid's binary_logloss: 0.0117461
    [45]	valid's binary_logloss: 0.0113894
    [46]	valid's binary_logloss: 0.0110523
    [47]	valid's binary_logloss: 0.0107424
    [48]	valid's binary_logloss: 0.0104404
    [49]	valid's binary_logloss: 0.0101549
    [50]	valid's binary_logloss: 0.00988521
    [51]	valid's binary_logloss: 0.00964179
    [52]	valid's binary_logloss: 0.00940021
    [53]	valid's binary_logloss: 0.00918043
    [54]	valid's binary_logloss: 0.008964
    [55]	valid's binary_logloss: 0.00875955
    [56]	valid's binary_logloss: 0.00856644
    [57]	valid's binary_logloss: 0.0083931
    [58]	valid's binary_logloss: 0.00822024
    [59]	valid's binary_logloss: 0.00805704
    [60]	valid's binary_logloss: 0.00791096
    [61]	valid's binary_logloss: 0.00776491
    [62]	valid's binary_logloss: 0.00762709
    [63]	valid's binary_logloss: 0.00749708
    [64]	valid's binary_logloss: 0.00738321
    [65]	valid's binary_logloss: 0.00726701
    [66]	valid's binary_logloss: 0.00715746
    [67]	valid's binary_logloss: 0.00706158
    [68]	valid's binary_logloss: 0.00696373
    [69]	valid's binary_logloss: 0.00687158
    [70]	valid's binary_logloss: 0.0067848
    [71]	valid's binary_logloss: 0.00670311
    [72]	valid's binary_logloss: 0.00663279
    [73]	valid's binary_logloss: 0.00656001
    [74]	valid's binary_logloss: 0.0064981
    [75]	valid's binary_logloss: 0.00643326
    [76]	valid's binary_logloss: 0.00637234
    [77]	valid's binary_logloss: 0.00632367
    [78]	valid's binary_logloss: 0.0062695
    [79]	valid's binary_logloss: 0.00621865
    [80]	valid's binary_logloss: 0.00617095
    [81]	valid's binary_logloss: 0.00612622
    [82]	valid's binary_logloss: 0.00609056
    [83]	valid's binary_logloss: 0.0060561
    [84]	valid's binary_logloss: 0.00601856
    [85]	valid's binary_logloss: 0.00598343
    [86]	valid's binary_logloss: 0.00595056
    [87]	valid's binary_logloss: 0.0059248
    [88]	valid's binary_logloss: 0.00590037
    [89]	valid's binary_logloss: 0.00587787
    [90]	valid's binary_logloss: 0.00585839
    [91]	valid's binary_logloss: 0.00583386
    [92]	valid's binary_logloss: 0.00581097
    [93]	valid's binary_logloss: 0.0057934
    [94]	valid's binary_logloss: 0.00577324
    [95]	valid's binary_logloss: 0.00575446
    [96]	valid's binary_logloss: 0.00573698
    [97]	valid's binary_logloss: 0.00572072
    [98]	valid's binary_logloss: 0.00570561
    [99]	valid's binary_logloss: 0.00569158
    [100]	valid's binary_logloss: 0.00567856
    [101]	valid's binary_logloss: 0.00566649
    [102]	valid's binary_logloss: 0.0056553
    [103]	valid's binary_logloss: 0.00564494
    [104]	valid's binary_logloss: 0.00563536
    [105]	valid's binary_logloss: 0.00562651
    [106]	valid's binary_logloss: 0.00561833
    [107]	valid's binary_logloss: 0.00561079
    [108]	valid's binary_logloss: 0.00560384
    [109]	valid's binary_logloss: 0.00559744
    [110]	valid's binary_logloss: 0.00559156
    [111]	valid's binary_logloss: 0.00558615
    [112]	valid's binary_logloss: 0.00558119
    [113]	valid's binary_logloss: 0.00557665
    [114]	valid's binary_logloss: 0.00557249
    [115]	valid's binary_logloss: 0.00556869
    [116]	valid's binary_logloss: 0.00556523
    [117]	valid's binary_logloss: 0.00556207
    [118]	valid's binary_logloss: 0.0055592
    [119]	valid's binary_logloss: 0.0055566
    [120]	valid's binary_logloss: 0.00555424
    [121]	valid's binary_logloss: 0.00555211
    [122]	valid's binary_logloss: 0.00555019
    [123]	valid's binary_logloss: 0.00554846
    [124]	valid's binary_logloss: 0.00554691
    [125]	valid's binary_logloss: 0.00554591
    [126]	valid's binary_logloss: 0.00554514
    [127]	valid's binary_logloss: 0.00554396
    [128]	valid's binary_logloss: 0.00554292
    [129]	valid's binary_logloss: 0.005542
    [130]	valid's binary_logloss: 0.00554175
    [131]	valid's binary_logloss: 0.00554139
    [132]	valid's binary_logloss: 0.00554073
    [133]	valid's binary_logloss: 0.00554016
    [134]	valid's binary_logloss: 0.00553967
    [135]	valid's binary_logloss: 0.00553926
    [136]	valid's binary_logloss: 0.00553892
    [137]	valid's binary_logloss: 0.00553865
    [138]	valid's binary_logloss: 0.00553854
    [139]	valid's binary_logloss: 0.00553836
    [140]	valid's binary_logloss: 0.00553853
    [141]	valid's binary_logloss: 0.00553844
    [142]	valid's binary_logloss: 0.00553838
    [143]	valid's binary_logloss: 0.00553836
    [144]	valid's binary_logloss: 0.00553838
    [145]	valid's binary_logloss: 0.00553854
    [146]	valid's binary_logloss: 0.0055386
    [147]	valid's binary_logloss: 0.00553889
    [148]	valid's binary_logloss: 0.005539
    [149]	valid's binary_logloss: 0.00553913
    Early stopping, best iteration is:
    [139]	valid's binary_logloss: 0.00553836
    [1]	valid's binary_logloss: 0.156172
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.155582
    [3]	valid's binary_logloss: 0.155
    [4]	valid's binary_logloss: 0.154421
    [5]	valid's binary_logloss: 0.153848
    [6]	valid's binary_logloss: 0.15328
    [7]	valid's binary_logloss: 0.152717
    [8]	valid's binary_logloss: 0.15216
    [9]	valid's binary_logloss: 0.151608
    [10]	valid's binary_logloss: 0.151061
    [11]	valid's binary_logloss: 0.150519
    [12]	valid's binary_logloss: 0.149982
    [13]	valid's binary_logloss: 0.149451
    [14]	valid's binary_logloss: 0.148924
    [15]	valid's binary_logloss: 0.148402
    [16]	valid's binary_logloss: 0.147885
    [17]	valid's binary_logloss: 0.147373
    [18]	valid's binary_logloss: 0.146864
    [19]	valid's binary_logloss: 0.146361
    [20]	valid's binary_logloss: 0.145862
    [21]	valid's binary_logloss: 0.145368
    [22]	valid's binary_logloss: 0.144878
    [23]	valid's binary_logloss: 0.144394
    [24]	valid's binary_logloss: 0.143916
    [25]	valid's binary_logloss: 0.143439
    [26]	valid's binary_logloss: 0.142966
    [27]	valid's binary_logloss: 0.142497
    [28]	valid's binary_logloss: 0.142032
    [29]	valid's binary_logloss: 0.141573
    [30]	valid's binary_logloss: 0.141115
    [31]	valid's binary_logloss: 0.140662
    [32]	valid's binary_logloss: 0.140212
    [33]	valid's binary_logloss: 0.139766
    [34]	valid's binary_logloss: 0.139324
    [35]	valid's binary_logloss: 0.138887
    [36]	valid's binary_logloss: 0.138453
    [37]	valid's binary_logloss: 0.138021
    [38]	valid's binary_logloss: 0.137594
    [39]	valid's binary_logloss: 0.137169
    [40]	valid's binary_logloss: 0.136748
    [41]	valid's binary_logloss: 0.136331
    [42]	valid's binary_logloss: 0.135917
    [43]	valid's binary_logloss: 0.135506
    [44]	valid's binary_logloss: 0.135099
    [45]	valid's binary_logloss: 0.134695
    [46]	valid's binary_logloss: 0.134293
    [47]	valid's binary_logloss: 0.133894
    [48]	valid's binary_logloss: 0.133499
    [49]	valid's binary_logloss: 0.133107
    [50]	valid's binary_logloss: 0.132717
    [51]	valid's binary_logloss: 0.132331
    [52]	valid's binary_logloss: 0.131947
    [53]	valid's binary_logloss: 0.131567
    [54]	valid's binary_logloss: 0.131189
    [55]	valid's binary_logloss: 0.130814
    [56]	valid's binary_logloss: 0.130442
    [57]	valid's binary_logloss: 0.130073
    [58]	valid's binary_logloss: 0.129706
    [59]	valid's binary_logloss: 0.129342
    [60]	valid's binary_logloss: 0.128981
    [61]	valid's binary_logloss: 0.128622
    [62]	valid's binary_logloss: 0.128265
    [63]	valid's binary_logloss: 0.127912
    [64]	valid's binary_logloss: 0.127561
    [65]	valid's binary_logloss: 0.127212
    [66]	valid's binary_logloss: 0.126865
    [67]	valid's binary_logloss: 0.126521
    [68]	valid's binary_logloss: 0.126179
    [69]	valid's binary_logloss: 0.12584
    [70]	valid's binary_logloss: 0.125503
    [71]	valid's binary_logloss: 0.125168
    [72]	valid's binary_logloss: 0.124836
    [73]	valid's binary_logloss: 0.124506
    [74]	valid's binary_logloss: 0.124178
    [75]	valid's binary_logloss: 0.123852
    [76]	valid's binary_logloss: 0.123528
    [77]	valid's binary_logloss: 0.123206
    [78]	valid's binary_logloss: 0.122888
    [79]	valid's binary_logloss: 0.12257
    [80]	valid's binary_logloss: 0.122255
    [81]	valid's binary_logloss: 0.121942
    [82]	valid's binary_logloss: 0.12163
    [83]	valid's binary_logloss: 0.121321
    [84]	valid's binary_logloss: 0.121013
    [85]	valid's binary_logloss: 0.120708
    [86]	valid's binary_logloss: 0.120405
    [87]	valid's binary_logloss: 0.120103
    [88]	valid's binary_logloss: 0.119804
    [89]	valid's binary_logloss: 0.119506
    [90]	valid's binary_logloss: 0.11921
    [91]	valid's binary_logloss: 0.118916
    [92]	valid's binary_logloss: 0.118624
    [93]	valid's binary_logloss: 0.118333
    [94]	valid's binary_logloss: 0.118044
    [95]	valid's binary_logloss: 0.117757
    [96]	valid's binary_logloss: 0.117471
    [97]	valid's binary_logloss: 0.117188
    [98]	valid's binary_logloss: 0.116905
    [99]	valid's binary_logloss: 0.116625
    [100]	valid's binary_logloss: 0.116346
    Did not meet early stopping. Best iteration is:
    [100]	valid's binary_logloss: 0.116346
    [1]	valid's binary_logloss: 0.156172
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.155581
    [3]	valid's binary_logloss: 0.154998
    [4]	valid's binary_logloss: 0.154417
    [5]	valid's binary_logloss: 0.153844
    [6]	valid's binary_logloss: 0.153275
    [7]	valid's binary_logloss: 0.152711
    [8]	valid's binary_logloss: 0.152153
    [9]	valid's binary_logloss: 0.1516
    [10]	valid's binary_logloss: 0.151052
    [11]	valid's binary_logloss: 0.15051
    [12]	valid's binary_logloss: 0.149972
    [13]	valid's binary_logloss: 0.14944
    [14]	valid's binary_logloss: 0.148912
    [15]	valid's binary_logloss: 0.14839
    [16]	valid's binary_logloss: 0.147873
    [17]	valid's binary_logloss: 0.147359
    [18]	valid's binary_logloss: 0.146851
    [19]	valid's binary_logloss: 0.146347
    [20]	valid's binary_logloss: 0.145848
    [21]	valid's binary_logloss: 0.145353
    [22]	valid's binary_logloss: 0.144863
    [23]	valid's binary_logloss: 0.144377
    [24]	valid's binary_logloss: 0.143899
    [25]	valid's binary_logloss: 0.143422
    [26]	valid's binary_logloss: 0.142948
    [27]	valid's binary_logloss: 0.142478
    [28]	valid's binary_logloss: 0.142014
    [29]	valid's binary_logloss: 0.141553
    [30]	valid's binary_logloss: 0.141094
    [31]	valid's binary_logloss: 0.14064
    [32]	valid's binary_logloss: 0.14019
    [33]	valid's binary_logloss: 0.139744
    [34]	valid's binary_logloss: 0.139302
    [35]	valid's binary_logloss: 0.138864
    [36]	valid's binary_logloss: 0.138429
    [37]	valid's binary_logloss: 0.137997
    [38]	valid's binary_logloss: 0.137569
    [39]	valid's binary_logloss: 0.137144
    [40]	valid's binary_logloss: 0.136723
    [41]	valid's binary_logloss: 0.136305
    [42]	valid's binary_logloss: 0.135891
    [43]	valid's binary_logloss: 0.135479
    [44]	valid's binary_logloss: 0.135071
    [45]	valid's binary_logloss: 0.134667
    [46]	valid's binary_logloss: 0.134265
    [47]	valid's binary_logloss: 0.133865
    [48]	valid's binary_logloss: 0.133469
    [49]	valid's binary_logloss: 0.133077
    [50]	valid's binary_logloss: 0.132688
    [51]	valid's binary_logloss: 0.132301
    [52]	valid's binary_logloss: 0.131916
    [53]	valid's binary_logloss: 0.131535
    [54]	valid's binary_logloss: 0.131157
    [55]	valid's binary_logloss: 0.130782
    [56]	valid's binary_logloss: 0.13041
    [57]	valid's binary_logloss: 0.13004
    [58]	valid's binary_logloss: 0.129673
    [59]	valid's binary_logloss: 0.129309
    [60]	valid's binary_logloss: 0.128947
    [61]	valid's binary_logloss: 0.128588
    [62]	valid's binary_logloss: 0.128231
    [63]	valid's binary_logloss: 0.127877
    [64]	valid's binary_logloss: 0.127526
    [65]	valid's binary_logloss: 0.127177
    [66]	valid's binary_logloss: 0.12683
    [67]	valid's binary_logloss: 0.126485
    [68]	valid's binary_logloss: 0.126144
    [69]	valid's binary_logloss: 0.125804
    [70]	valid's binary_logloss: 0.125466
    [71]	valid's binary_logloss: 0.125131
    [72]	valid's binary_logloss: 0.124798
    [73]	valid's binary_logloss: 0.124468
    [74]	valid's binary_logloss: 0.12414
    [75]	valid's binary_logloss: 0.123814
    [76]	valid's binary_logloss: 0.12349
    [77]	valid's binary_logloss: 0.123168
    [78]	valid's binary_logloss: 0.122849
    [79]	valid's binary_logloss: 0.122531
    [80]	valid's binary_logloss: 0.122215
    [81]	valid's binary_logloss: 0.121901
    [82]	valid's binary_logloss: 0.12159
    [83]	valid's binary_logloss: 0.12128
    [84]	valid's binary_logloss: 0.120972
    [85]	valid's binary_logloss: 0.120666
    [86]	valid's binary_logloss: 0.120363
    [87]	valid's binary_logloss: 0.120061
    [88]	valid's binary_logloss: 0.119761
    [89]	valid's binary_logloss: 0.119463
    [90]	valid's binary_logloss: 0.119166
    [91]	valid's binary_logloss: 0.118872
    [92]	valid's binary_logloss: 0.118579
    [93]	valid's binary_logloss: 0.118289
    [94]	valid's binary_logloss: 0.118
    [95]	valid's binary_logloss: 0.117712
    [96]	valid's binary_logloss: 0.117426
    [97]	valid's binary_logloss: 0.117142
    [98]	valid's binary_logloss: 0.11686
    [99]	valid's binary_logloss: 0.116579
    [100]	valid's binary_logloss: 0.1163
    Did not meet early stopping. Best iteration is:
    [100]	valid's binary_logloss: 0.1163
    [1]	valid's binary_logloss: 0.156172
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.155583
    [3]	valid's binary_logloss: 0.155
    [4]	valid's binary_logloss: 0.154421
    [5]	valid's binary_logloss: 0.153849
    [6]	valid's binary_logloss: 0.15328
    [7]	valid's binary_logloss: 0.152718
    [8]	valid's binary_logloss: 0.15216
    [9]	valid's binary_logloss: 0.151608
    [10]	valid's binary_logloss: 0.151061
    [11]	valid's binary_logloss: 0.15052
    [12]	valid's binary_logloss: 0.149983
    [13]	valid's binary_logloss: 0.149451
    [14]	valid's binary_logloss: 0.148924
    [15]	valid's binary_logloss: 0.148403
    [16]	valid's binary_logloss: 0.147886
    [17]	valid's binary_logloss: 0.147374
    [18]	valid's binary_logloss: 0.146865
    [19]	valid's binary_logloss: 0.146362
    [20]	valid's binary_logloss: 0.145864
    [21]	valid's binary_logloss: 0.145371
    [22]	valid's binary_logloss: 0.144881
    [23]	valid's binary_logloss: 0.144396
    [24]	valid's binary_logloss: 0.143918
    [25]	valid's binary_logloss: 0.143441
    [26]	valid's binary_logloss: 0.142968
    [27]	valid's binary_logloss: 0.142499
    [28]	valid's binary_logloss: 0.142034
    [29]	valid's binary_logloss: 0.141574
    [30]	valid's binary_logloss: 0.141116
    [31]	valid's binary_logloss: 0.140663
    [32]	valid's binary_logloss: 0.140213
    [33]	valid's binary_logloss: 0.139767
    [34]	valid's binary_logloss: 0.139325
    [35]	valid's binary_logloss: 0.138887
    [36]	valid's binary_logloss: 0.138453
    [37]	valid's binary_logloss: 0.138022
    [38]	valid's binary_logloss: 0.137594
    [39]	valid's binary_logloss: 0.13717
    [40]	valid's binary_logloss: 0.136749
    [41]	valid's binary_logloss: 0.136332
    [42]	valid's binary_logloss: 0.135918
    [43]	valid's binary_logloss: 0.135507
    [44]	valid's binary_logloss: 0.135099
    [45]	valid's binary_logloss: 0.134695
    [46]	valid's binary_logloss: 0.134293
    [47]	valid's binary_logloss: 0.133895
    [48]	valid's binary_logloss: 0.133499
    [49]	valid's binary_logloss: 0.133107
    [50]	valid's binary_logloss: 0.132718
    [51]	valid's binary_logloss: 0.132331
    [52]	valid's binary_logloss: 0.131948
    [53]	valid's binary_logloss: 0.131567
    [54]	valid's binary_logloss: 0.131189
    [55]	valid's binary_logloss: 0.130814
    [56]	valid's binary_logloss: 0.130443
    [57]	valid's binary_logloss: 0.130074
    [58]	valid's binary_logloss: 0.129707
    [59]	valid's binary_logloss: 0.129343
    [60]	valid's binary_logloss: 0.128981
    [61]	valid's binary_logloss: 0.128623
    [62]	valid's binary_logloss: 0.128266
    [63]	valid's binary_logloss: 0.127912
    [64]	valid's binary_logloss: 0.127561
    [65]	valid's binary_logloss: 0.127212
    [66]	valid's binary_logloss: 0.126865
    [67]	valid's binary_logloss: 0.126522
    [68]	valid's binary_logloss: 0.12618
    [69]	valid's binary_logloss: 0.125841
    [70]	valid's binary_logloss: 0.125504
    [71]	valid's binary_logloss: 0.125169
    [72]	valid's binary_logloss: 0.124837
    [73]	valid's binary_logloss: 0.124507
    [74]	valid's binary_logloss: 0.124179
    [75]	valid's binary_logloss: 0.123853
    [76]	valid's binary_logloss: 0.123529
    [77]	valid's binary_logloss: 0.123208
    [78]	valid's binary_logloss: 0.122889
    [79]	valid's binary_logloss: 0.122571
    [80]	valid's binary_logloss: 0.122256
    [81]	valid's binary_logloss: 0.121942
    [82]	valid's binary_logloss: 0.121631
    [83]	valid's binary_logloss: 0.121322
    [84]	valid's binary_logloss: 0.121014
    [85]	valid's binary_logloss: 0.120709
    [86]	valid's binary_logloss: 0.120405
    [87]	valid's binary_logloss: 0.120104
    [88]	valid's binary_logloss: 0.119804
    [89]	valid's binary_logloss: 0.119507
    [90]	valid's binary_logloss: 0.119211
    [91]	valid's binary_logloss: 0.118917
    [92]	valid's binary_logloss: 0.118624
    [93]	valid's binary_logloss: 0.118333
    [94]	valid's binary_logloss: 0.118044
    [95]	valid's binary_logloss: 0.117757
    [96]	valid's binary_logloss: 0.117471
    [97]	valid's binary_logloss: 0.117187
    [98]	valid's binary_logloss: 0.116905
    [99]	valid's binary_logloss: 0.116625
    [100]	valid's binary_logloss: 0.116346
    Did not meet early stopping. Best iteration is:
    [100]	valid's binary_logloss: 0.116346


    [Parallel(n_jobs=1)]: Done 600 out of 600 | elapsed: 37.3min finished


    [1]	valid's binary_logloss: 0.110544
    Training until validation scores don't improve for 10 rounds.
    [2]	valid's binary_logloss: 0.0970173
    [3]	valid's binary_logloss: 0.0874035
    [4]	valid's binary_logloss: 0.0797989
    [5]	valid's binary_logloss: 0.0735331
    [6]	valid's binary_logloss: 0.0681552
    [7]	valid's binary_logloss: 0.0634505
    [8]	valid's binary_logloss: 0.0592829
    [9]	valid's binary_logloss: 0.0555562
    [10]	valid's binary_logloss: 0.0521827
    [11]	valid's binary_logloss: 0.049132
    [12]	valid's binary_logloss: 0.0463235
    [13]	valid's binary_logloss: 0.0437418
    [14]	valid's binary_logloss: 0.0413624
    [15]	valid's binary_logloss: 0.0391784
    [16]	valid's binary_logloss: 0.0371359
    [17]	valid's binary_logloss: 0.0352444
    [18]	valid's binary_logloss: 0.0334745
    [19]	valid's binary_logloss: 0.0318232
    [20]	valid's binary_logloss: 0.0302783
    [21]	valid's binary_logloss: 0.0288448
    [22]	valid's binary_logloss: 0.0275023
    [23]	valid's binary_logloss: 0.0262317
    [24]	valid's binary_logloss: 0.025045
    [25]	valid's binary_logloss: 0.0239328
    [26]	valid's binary_logloss: 0.0228786
    [27]	valid's binary_logloss: 0.021886
    [28]	valid's binary_logloss: 0.0209549
    [29]	valid's binary_logloss: 0.0200748
    [30]	valid's binary_logloss: 0.0192538
    [31]	valid's binary_logloss: 0.0184811
    [32]	valid's binary_logloss: 0.0177472
    [33]	valid's binary_logloss: 0.0170617
    [34]	valid's binary_logloss: 0.0164039
    [35]	valid's binary_logloss: 0.0157831
    [36]	valid's binary_logloss: 0.0151973
    [37]	valid's binary_logloss: 0.0146433
    [38]	valid's binary_logloss: 0.0141207
    [39]	valid's binary_logloss: 0.0136333
    [40]	valid's binary_logloss: 0.0131666
    [41]	valid's binary_logloss: 0.0127328
    [42]	valid's binary_logloss: 0.0123157
    [43]	valid's binary_logloss: 0.0119217
    [44]	valid's binary_logloss: 0.0115484
    [45]	valid's binary_logloss: 0.0111956
    [46]	valid's binary_logloss: 0.0108687
    [47]	valid's binary_logloss: 0.0105539
    [48]	valid's binary_logloss: 0.0102565
    [49]	valid's binary_logloss: 0.00998143
    [50]	valid's binary_logloss: 0.0097148
    [51]	valid's binary_logloss: 0.00947157
    [52]	valid's binary_logloss: 0.00923412
    [53]	valid's binary_logloss: 0.00901655
    [54]	valid's binary_logloss: 0.00881212
    [55]	valid's binary_logloss: 0.00861041
    [56]	valid's binary_logloss: 0.00841997
    [57]	valid's binary_logloss: 0.00824019
    [58]	valid's binary_logloss: 0.0080705
    [59]	valid's binary_logloss: 0.00792365
    [60]	valid's binary_logloss: 0.00777322
    [61]	valid's binary_logloss: 0.00763025
    [62]	valid's binary_logloss: 0.00749961
    [63]	valid's binary_logloss: 0.00737817
    [64]	valid's binary_logloss: 0.00726418
    [65]	valid's binary_logloss: 0.0071554
    [66]	valid's binary_logloss: 0.00705189
    [67]	valid's binary_logloss: 0.00695089
    [68]	valid's binary_logloss: 0.00685583
    [69]	valid's binary_logloss: 0.00677121
    [70]	valid's binary_logloss: 0.00668699
    [71]	valid's binary_logloss: 0.00661267
    [72]	valid's binary_logloss: 0.00653796
    [73]	valid's binary_logloss: 0.00647002
    [74]	valid's binary_logloss: 0.00641091
    [75]	valid's binary_logloss: 0.0063555
    [76]	valid's binary_logloss: 0.00629686
    [77]	valid's binary_logloss: 0.0062419
    [78]	valid's binary_logloss: 0.00619367
    [79]	valid's binary_logloss: 0.00614528
    [80]	valid's binary_logloss: 0.0061
    [81]	valid's binary_logloss: 0.00606381
    [82]	valid's binary_logloss: 0.006024
    [83]	valid's binary_logloss: 0.00599037
    [84]	valid's binary_logloss: 0.00595918
    [85]	valid's binary_logloss: 0.00593232
    [86]	valid's binary_logloss: 0.00590688
    [87]	valid's binary_logloss: 0.00588193
    [88]	valid's binary_logloss: 0.00585744
    [89]	valid's binary_logloss: 0.00583242
    [90]	valid's binary_logloss: 0.00580919
    [91]	valid's binary_logloss: 0.00579061
    [92]	valid's binary_logloss: 0.00577046
    [93]	valid's binary_logloss: 0.0057564
    [94]	valid's binary_logloss: 0.00574313
    [95]	valid's binary_logloss: 0.00573146
    [96]	valid's binary_logloss: 0.00571613
    [97]	valid's binary_logloss: 0.00570653
    [98]	valid's binary_logloss: 0.00569323
    [99]	valid's binary_logloss: 0.00568647
    [100]	valid's binary_logloss: 0.00567869
    [101]	valid's binary_logloss: 0.00566805
    [102]	valid's binary_logloss: 0.00565703
    [103]	valid's binary_logloss: 0.00565029
    [104]	valid's binary_logloss: 0.00564634
    [105]	valid's binary_logloss: 0.00563852
    [106]	valid's binary_logloss: 0.00563374
    [107]	valid's binary_logloss: 0.00563004
    [108]	valid's binary_logloss: 0.00562564
    [109]	valid's binary_logloss: 0.00562183
    [110]	valid's binary_logloss: 0.00561828
    [111]	valid's binary_logloss: 0.00561533
    [112]	valid's binary_logloss: 0.00561245
    [113]	valid's binary_logloss: 0.00560938
    [114]	valid's binary_logloss: 0.00560207
    [115]	valid's binary_logloss: 0.00560022
    [116]	valid's binary_logloss: 0.00559891
    [117]	valid's binary_logloss: 0.00559694
    [118]	valid's binary_logloss: 0.00559461
    [119]	valid's binary_logloss: 0.00559508
    [120]	valid's binary_logloss: 0.00559601
    [121]	valid's binary_logloss: 0.00559419
    [122]	valid's binary_logloss: 0.00558848
    [123]	valid's binary_logloss: 0.00558384
    [124]	valid's binary_logloss: 0.00558205
    [125]	valid's binary_logloss: 0.0055775
    [126]	valid's binary_logloss: 0.00557707
    [127]	valid's binary_logloss: 0.00557351
    [128]	valid's binary_logloss: 0.00557002
    [129]	valid's binary_logloss: 0.00557169
    [130]	valid's binary_logloss: 0.00556897
    [131]	valid's binary_logloss: 0.00556894
    [132]	valid's binary_logloss: 0.00556901
    [133]	valid's binary_logloss: 0.00556648
    [134]	valid's binary_logloss: 0.00556701
    [135]	valid's binary_logloss: 0.00556528
    [136]	valid's binary_logloss: 0.00556389
    [137]	valid's binary_logloss: 0.0055619
    [138]	valid's binary_logloss: 0.0055604
    [139]	valid's binary_logloss: 0.00555884
    [140]	valid's binary_logloss: 0.00555775
    [141]	valid's binary_logloss: 0.00555643
    [142]	valid's binary_logloss: 0.00555838
    [143]	valid's binary_logloss: 0.00555766
    [144]	valid's binary_logloss: 0.00555637
    [145]	valid's binary_logloss: 0.00555726
    [146]	valid's binary_logloss: 0.00555604
    [147]	valid's binary_logloss: 0.00555557
    [148]	valid's binary_logloss: 0.00555475
    [149]	valid's binary_logloss: 0.00555419
    [150]	valid's binary_logloss: 0.00555342
    [151]	valid's binary_logloss: 0.00555279
    [152]	valid's binary_logloss: 0.00555184
    [153]	valid's binary_logloss: 0.00555112
    [154]	valid's binary_logloss: 0.00555061
    [155]	valid's binary_logloss: 0.00555028
    [156]	valid's binary_logloss: 0.0055501
    [157]	valid's binary_logloss: 0.00555028
    [158]	valid's binary_logloss: 0.00555053
    [159]	valid's binary_logloss: 0.00554996
    [160]	valid's binary_logloss: 0.00554982
    [161]	valid's binary_logloss: 0.00554939
    [162]	valid's binary_logloss: 0.00554934
    [163]	valid's binary_logloss: 0.00555027
    [164]	valid's binary_logloss: 0.00555213
    [165]	valid's binary_logloss: 0.00555361
    [166]	valid's binary_logloss: 0.00555446
    [167]	valid's binary_logloss: 0.00555481
    [168]	valid's binary_logloss: 0.00555506
    [169]	valid's binary_logloss: 0.00555479
    [170]	valid's binary_logloss: 0.00555389
    [171]	valid's binary_logloss: 0.00555369
    [172]	valid's binary_logloss: 0.00555424
    Early stopping, best iteration is:
    [162]	valid's binary_logloss: 0.00554934
    Best score reached: 0.9991718806792841 with params: {'colsample_bytree': 0.44977115332798534, 'learning_rate': 0.05, 'max_depth': 2, 'min_child_samples': 70, 'min_child_weight': 0.1, 'n_estimators': 1000, 'num_leaves': 11, 'reg_alpha': 0, 'reg_lambda': 1, 'subsample': 0.7468010674479191} 
    -------------------Timing: 2239.568426847458 ------------------


$\implies$ Results of the Tuninng : 
  * Best score reached:
           0.9991718806792841 
  * Best  params:
  
         {'colsample_bytree': 0.44977115332798534, 'learning_rate': 0.05, 'max_depth': 2, 'min_child_samples': 70, 'min_child_weight': 0.1, 'n_estimators': 1000, 'num_leaves': 11, 'reg_alpha': 0, 'reg_lambda': 1, 'subsample': 0.7468010674479191} 


##### Meta classifier LGBM fitting using the optimal hyperprameters &  Predicting the final output for the Test Set (To be submitted ) : 



```python
opt_parameters= {'colsample_bytree': 0.44977115332798534, 'learning_rate': 0.05, 'max_depth': 2, 'min_child_samples': 70, 'min_child_weight': 0.1, 'n_estimators': 1000, 'num_leaves': 11, 'reg_alpha': 0, 'reg_lambda': 1, 'subsample': 0.7468010674479191} 
start_time=time.time()
lgbm_lvl_2 = lgb.LGBMClassifier()
lgbm_lvl_2.set_params(**opt_parameters)
lgbm_lvl_2.fit(X_stack_train,y_stack_train)
print("Timming:",time.time()-start_time)
prediction_val = lgbm_lvl_2.predict(X_stack_test)
print("Timming:",time.time()-start_time)
prediction_test = lgbm_lvl_2.predict(stacking_test_df)
scores(np.rint(prediction_val),y_stack_test)
print("Timming:",time.time()-start_time)
np.savetxt('LGBM_Stack_Sub.csv', prediction_test, fmt = '%1.0d', delimiter=',')
files.download('LGBM_Stack_Sub.csv')
```

    Timming: 6.396286487579346
    Timming: 8.794532299041748
    ========================================
    Classification Accuraccy:  0.9989109720
    precision score :  0.9818351188
    recall score:  0.9882794187
    f1 score:  0.9850467290
    ========================================
    Timming: 52.99763464927673


####XGBoost

##### Meta classifier XGB hyperprameter Tunning: 


```python
start_time=time.time()
xg = xgb.XGBClassifier()
xg_params= {'eta': [0.01,0.05,0.1,0.2,0.3],
             'min_depth': np.arange(3,10,1),
             'min_child_weight': np.arange(1,6,1),
             'scale_pos_weight': [0.5,1,2],
             'objective': ['binary:logistic', 'binary:logitraw','binary:hinge']
            }

# we tune out model on The output of the base level
xg_rg ,acc= random_grid(xg,xg_params,3,360,'accuracy',X_stack_train, y_stack_train,X_stack_test,y_stack_test) 
print("------------Tunning Time {} s------------ ".format(time.time()-start_time))
print(xg_rg,acc)
```

    Fitting 3 folds for each of 360 candidates, totalling 1080 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 2 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  28 tasks      | elapsed:  1.6min
    [Parallel(n_jobs=-1)]: Done 124 tasks      | elapsed:  7.0min
    [Parallel(n_jobs=-1)]: Done 284 tasks      | elapsed: 15.7min
    [Parallel(n_jobs=-1)]: Done 508 tasks      | elapsed: 27.9min
    [Parallel(n_jobs=-1)]: Done 796 tasks      | elapsed: 43.5min
    [Parallel(n_jobs=-1)]: Done 1080 out of 1080 | elapsed: 58.9min finished


    Training Best Score:  0.9991775528933727 
    
    Training Best Params:  
     {'scale_pos_weight': 1, 'objective': 'binary:logitraw', 'min_depth': 9, 'min_child_weight': 4, 'eta': 0.2} 
    
    
    Training Best Estimator:  
     XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=1, eta=0.2, gamma=0,
                  learning_rate=0.1, max_delta_step=0, max_depth=3,
                  min_child_weight=4, min_depth=9, missing=None, n_estimators=100,
                  n_jobs=1, nthread=None, objective='binary:logitraw',
                  random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                  seed=None, silent=None, subsample=1, verbosity=1) 
    
    
    ------------Tunning Time 3539.579792022705 s------------ 
    {'scale_pos_weight': 1, 'objective': 'binary:logitraw', 'min_depth': 9, 'min_child_weight': 4, 'eta': 0.2} 0.9991775528933727


$\implies$ Results of the Tuninng : 
  * Best score reached:
           0.9991775528933727 
  * Best  params:
  
         {'scale_pos_weight': 1, 'objective': 'binary:logitraw', 'min_depth': 9, 'min_child_weight': 4, 'eta': 0.2}

##### Meta classifier XGB fitting using the optimal hyperprameters &  Predicting the The final output for the Test Set (To be submitted ) : 



```python
opt_parameters={'scale_pos_weight': 1, 'objective': 'binary:logitraw', 'min_depth': 9, 'min_child_weight': 4, 'eta': 0.2} 
start_time=time.time()
xg_lvl_2 = xgb.XGBClassifier()
xg_lvl_2.set_params(**opt_parameters)
xg_lvl_2.fit(X_stack_train,y_stack_train)
print("Timming:",time.time()-start_time)
prediction_val = xg_lvl_2.predict(X_stack_test)
print("Timming:",time.time()-start_time)
prediction_test = xg_lvl_2.predict(stacking_test_df[X_stack_train.columns])
scores(np.rint(prediction_val),y_stack_test)
print("Timming:",time.time()-start_time)
np.savetxt('XGB_Stack_Sub.csv', prediction_test, fmt = '%1.0d', delimiter=',')
files.download('XGB_Stack_Sub.csv')
```

    Timming: 6.4949774742126465
    Timming: 6.660034418106079
    ========================================
    Classification Accuraccy:  0.9989393321
    precision score :  0.9818351188
    recall score:  0.9890522365
    f1 score:  0.9854304636
    ========================================
    Timming: 9.969765901565552


#Results: 



Results of the MetaClassifier: 



  * LightGBM : 

    * On the Training Set : Accuracy:   0.9991718806792841  

    * On our Testing Set : Accuracy:  0.9989393321

    * On The challange Test Set (from the website) : Accuracy 0.998666182884


  * XGBoost  : 

    * On the Training Set : Accuracy:  0.9991775528933727 

    * On our Testing Set :  Accuracy:  0.9989393321

    * On The challange Test Set (from the website) :0.99864258884


$\implies$ The ensemble method (Stacking) improved my results, especially after getting stuck at some constant accuracy 0.9985 using single models.

$\implies$ Stacking the models improved the results slightly because it combined the output of all the models  (some models managed to identify some observations while other models failed to do so, so stacking made the models complementary, Thus improved the results.

#Conclusion 

* Some of Exploratory data Analysis techniques really impacted the results of our models, like the use of resampling and outliers elimination.

* Tuning the hyperparameter also made a huge impact on the performances of some algorithms and boosted the results provided by those algorithms.

* The use of ensemble methods like stacking in classification  by training diverse and accurate classifiers, Diversity can be achieved by varying  hyper-parameter settings, have been very successful in setting record performance compared to single models.


```python

```
