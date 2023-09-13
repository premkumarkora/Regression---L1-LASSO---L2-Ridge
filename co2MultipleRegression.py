from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("../../data/co2Emmission.csv")

#sns.pairplot(df,hue="CO2")
sns.pairplot(df)
plt.show()

#print(df)
x = df[['Weight', 'Volume']]
y = df[['CO2']]
#print(x,y)

x_scaled = preprocessing.scale(x)
x_scaled = pd.DataFrame(x_scaled, columns=x.columns)

y_scaled = preprocessing.scale(y)
y_scaled = pd.DataFrame(y_scaled, columns=y.columns)


x_train,x_test,y_train,y_test =train_test_split(x,y,train_size=0.8)
regressor = LinearRegression()
regressor.fit(x_train,y_train )
#print(x_train)
yTrainPred = regressor.predict(x_train)
yTestPred = regressor.predict(x_test)

for idx, col_name in enumerate(x_train.columns):
    print("The Coefficent for {} is {}".format(col_name, regressor.coef_[0][idx]))
intercept = regressor.intercept_[0]
print("The intercept for model is {}".format(intercept))

ridge= Ridge(alpha=0.9)
ridge.fit(x_train,y_train)
for i, col_name in enumerate(x_train.columns):
    print("The Coefficent for Ridge {} is {}".format(col_name, ridge.coef_[0][i]))

lasso= Lasso(alpha=0.1)
lasso.fit(x_train,y_train)
for li, col in enumerate(x_train):
    print("The Coefficent for Lasso {} is {}".format(col, lasso.coef_[li]))



'''
Volume = int(input(" Please Enter Volume :"))
Weight = int(input(" Please Enter Weight :"))
predictCO2 = regressor.predict([[Volume, Weight]])
print("Volume of CO2 :", *predictCO2[0])
print("Regressor Coeffecient :", regressor.coef_)
print("Regressor Score for Training Data:",regressor.score(x_train,y_train))
print("Regressor Score for Test Data:",regressor.score(x_test,y_test))

'''