# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 18:15:31 2022

@author: Muqeet
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.random import randint

data=pd.read_csv("dataPrice_Assignment.csv")

data.head()

data.columns

data.shape

data.info()

data.describe()


data.isnull().sum()

sns.heatmap(data.isnull())

#categorical features

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data['CarName']=le.fit_transform(data['CarName'])
data['CarName']=data['CarName'].astype(float)

data['fueltype']=le.fit_transform(data['fueltype'])
data['fueltype']=data['fueltype'].astype(float)

data['aspiration']=le.fit_transform(data['aspiration'])
data['aspiration']=data['aspiration'].astype(float)

data['doornumber']=le.fit_transform(data['doornumber'])
data['doornumber']=data['doornumber'].astype(float)

data['carbody']=le.fit_transform(data['carbody'])
data['carbody']=data['carbody'].astype(float)

data['drivewheel']=le.fit_transform(data['drivewheel'])
data['drivewheel']=data['drivewheel'].astype(float)

data['enginelocation']=le.fit_transform(data['enginelocation'])
data['enginelocation']=data['enginelocation'].astype(float)

data['enginetype']=le.fit_transform(data['enginetype'])
data['enginetype']=data['enginetype'].astype(float)

data['cylindernumber'] = le.fit_transform(data['cylindernumber'])
data['cylindernumber'] = data['cylindernumber'].astype(float)

data['fuelsystem'] = le.fit_transform(data['fuelsystem'])
data['fuelsystem'] = data['fuelsystem'].astype(float)


data.info()


a=data.corr()


plt.figure(figsize=(15,7))
sns.heatmap(data.corr(),annot=True,cmap="Blues")
plt.title("data corelation",size=15)
plt.show()

data['price'].describe()

sns.distplot(data['price'])

round(data['fueltype'].value_counts()/data.shape[0]*100,2)


round(data['fueltype'].value_counts()/data.shape[0]*100,2).plot.pie(autopct='%1.1f%%')


sns.barplot(x=data['fueltype'],y=data['price'])

#splitdata


data.columns

x=data.drop(['price'],axis=1).values
y=data['price'].values



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)


x_train


#featurescaling using robust scaller

from sklearn.preprocessing import RobustScaler

rs=RobustScaler()

x_train=rs.fit_transform(x_train)
x_test=rs.fit_transform(x_test)

x_train.shape

#multiple linear regression

from sklearn import linear_model 
reg=linear_model.LinearRegression()

reg.fit(x_train,y_train)

reg.coef_

pd.DataFrame(reg.coef_ , data.columns[:-1] ,  columns=['Coeficient'])


y_pred=reg.predict(x_test)


df=pd.DataFrame({"y_test":y_test,"y_pred":y_pred})

df.head(10)
df.shape


#lesso regression

las=linear_model.Lasso(alpha=0.9)

las.fit(x_train,y_train)


las.score(x_test,y_test)

y_pred1=las.predict(x_test)