
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("../../data/auto-mpg.csv")
#print(data.shape)
#print(data.info())
#print(data.isnull().sum())
#print(data[data.horsepower.str.isdigit()==False])
data["horsepower"] = data["horsepower"].replace('?',np.nan)
#print(data.isnull().sum())
data["horsepower"] = data["horsepower"].fillna(data["horsepower"].median())
#print(data.isnull().sum())
#print(data.dtypes)
data["horsepower"] = data["horsepower"].astype("float64")

# To see the relation between all the columns for the given output column
#sns.pairplot(data[["mpg","cylinders","displacement","horsepower","weight","acceleration","model year","origin"]])
#plt.show()

#print(data.corr().T) # There are 4 negative corelation

data= data.drop("car name", axis=1) # Remove the car name Column
#print(data.head())

x=data.drop("mpg", axis=1)
y=data[["mpg"]]
print(y.value_counts())
#scale the data, This will give numpy array
x_scaled = preprocessing.scale(x)
x_scaled = pd.DataFrame(x_scaled, columns=x.columns)

y_scaled = preprocessing.scale(y)
y_scaled = pd.DataFrame(y_scaled, columns=y.columns)
print(y_scaled.value_counts())
x_train, x_test,y_train,y_test = train_test_split(x_scaled,y_scaled,test_size=0.30, random_state=1)
regressor = LinearRegression()
regressor.fit(x_train,y_train )
'''
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.30, random_state=1)
regressor = LinearRegression()
regressor.fit(x_train,y_train )
'''
for idx, col_name in enumerate(x_train.columns):
    print("The Coefficent for {} is {}".format(col_name, regressor.coef_[0][idx]))
intercept = regressor.intercept_[0]
print("The intercept for model is {}".format(intercept))




ridge= Ridge(alpha=0.3)
ridge.fit(x_train,y_train)

for i, col_name in enumerate(x_train.columns):
    print("The Coefficent for Ridge {} is {}".format(col_name, ridge.coef_[0][i]))
interceptR = ridge.intercept_[0]
print("The intercept for Ridge model is {}".format(interceptR))
print("-------------------------------------------------------")
lasso= Lasso(alpha=0.1)
lasso.fit(x_train,y_train)

for li, col in enumerate(x_train):
    print("The Coefficent for Lasso {} is {}".format(col, lasso.coef_[li]))
#Comparing the score
print("Linear Regression Training Data:",regressor.score(x_train,y_train))
print("Linear Regression Testing Data:",regressor.score(x_test,y_test))
print("Linear ridge Training Data:",ridge.score(x_train,y_train))
print("Linear ridge Testing Data:",ridge.score(x_test,y_test))
print("Linear lasso Training Data:",lasso.score(x_train,y_train))
print("Linear lasso Testing Data:",lasso.score(x_test,y_test))