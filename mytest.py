
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 00:46:40 2019

@author: HARSH
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt


salary = pd.read_csv("Salary_Data.csv")
x=  salary.iloc[:,:-1]
y = salary.iloc[:,1:2]
for i in range(1,30):   
    for j in range(1,5):
        from sklearn.cross_validation import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(x,y,train_size = i/30,random_state = j)
        
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(X_train,y_train)
        pred_y = regressor.predict(X_test)
        plt.scatter(X_train,y_train,color="red")
        plt.plot(X_train,regressor.predict(X_train),color = "green")
        plt.scatter(X_test,y_test,color = "blue")
        plt.show()
