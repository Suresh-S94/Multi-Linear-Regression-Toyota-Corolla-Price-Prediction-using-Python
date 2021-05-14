import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

toyota=pd.read_csv("E:\\Data Science_Excelr\\Multi Linear Regression\\ToyotaCorolla.csv", encoding='latin1')

toyota1= toyota.iloc[:,[2,3,6,8,12,13,15,16,17]]
toyota1.rename(columns={"Age_08_04":"Age"},inplace=True)

eda=toyota1.describe()

eda

plt.boxplot(toyota1["Price"])
plt.boxplot(toyota1["Age"])
plt.boxplot(toyota1["HP"])
plt.boxplot(toyota1["cc"])
plt.boxplot(toyota1["Quarterly_Tax"])
plt.boxplot(toyota1["Weight"])

import statsmodels.api as sm
sm.graphics.qqplot(toyota1["Price"]) ##shows the data "Price" is not normal 
sm.graphics.qqplot(toyota1["Age"])## shows "Age" is not normal. Data is discrete count 
sm.graphics.qqplot(toyota1["HP"])## Data is discrete count
sm.graphics.qqplot(toyota1["Quarterly_Tax"]) ## Data is discrete count
sm.graphics.qqplot(toyota1["Weight"]) ## Data is not normal. And it shows it is discrete count
sm.graphics.qqplot(toyota1["Gears"]) ## Data is discrete categorical
sm.graphics.qqplot(toyota1["Doors"]) ## Data is discrete categorical 
sm.graphics.qqplot(toyota1["cc"]) ## Again data is discrete count data.

plt.hist(toyota1["Price"]) ## This shows that Price is right skewed

plt.hist(toyota1["Age"])

plt.hist(toyota1["HP"])## The data is very unevenly distributed, Left skewed

plt.hist(toyota1["Quarterly_Tax"]) # The data is unevenly distributed, right skewed data

plt.hist(toyota1["Weight"]) # The data is right skewed.

import seaborn as sn
sn.pairplot(toyota1)
correlation_values= toyota1.corr()

correlation_values

import statsmodels.formula.api as smf
m1= smf.ols("Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight",data= toyota1).fit()
m1.summary()

m1_cc = smf.ols("Price~cc",data= toyota1).fit()
m1_cc.summary()

m1_doors = smf.ols("Price~Doors", data= toyota1).fit()
m1_doors.summary()

m1_to = smf.ols("Price~cc+Doors",data= toyota1).fit()
m1_to.summary()

import statsmodels.api as sm
sm.graphics.influence_plot(m1)

toyota2= toyota1.drop(toyota.index[[80]],axis=0)
m2= smf.ols("Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight",data= toyota2).fit()
m2.summary()

toyota3 = toyota1.drop(toyota.index[[80,221]],axis=0)

m3= smf.ols("Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight",data= toyota3).fit()
m3.summary()

toyota4= toyota1.drop(toyota.index[[80,221,960]],axis=0)

m4= smf.ols("Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight", data = toyota4).fit()
m4.summary()

finalmodel = smf.ols("Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight", data = toyota4).fit()
finalmodel.summary()

finalmodel_pred = finalmodel.predict(toyota4)

plt.scatter(toyota4["Price"],finalmodel_pred,c='r');plt.xlabel("Observed values");plt.ylabel("Predicted values")

plt.scatter(finalmodel_pred, finalmodel.resid_pearson,c='r');plt.axhline(y=0,color='blue');plt.xlabel("Fitted values");plt.ylabel("Residuals")

plt.hist(finalmodel.resid_pearson)

import pylab
import scipy.stats as st
st.probplot(finalmodel.resid_pearson, dist='norm',plot=pylab)

from sklearn.model_selection import train_test_split

train_data,test_Data= train_test_split(toyota1,test_size=0.3)

finalmodel1 = smf.ols("Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight", data = train_data).fit()
finalmodel1.summary()

finalmodel_pred = finalmodel1.predict(train_data)
finalmodel_pred

finalmodel_res = train_data["Price"]-finalmodel_pred
finalmodel_res

finalmodel_rmse = np.sqrt(np.mean(finalmodel_res*finalmodel_res))
finalmodel_rmse

finalmodel_testpred = finalmodel1.predict(test_Data)
finalmodel_testpred

finalmodel_testres= test_Data["Price"]-finalmodel_testpred
finalmodel_testres

finalmodel_testrmse = np.sqrt(np.mean(finalmodel_testres*finalmodel_testres))
finalmodel_testrmse

