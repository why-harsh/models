By using our site, you acknowledge that you have read and understand our Cookie Policy, Privacy Policy, and our Terms of Service.

 
# coding: utf-8

# # Preparing the Data

# In[79]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[134]:


#Data for Rendement 20  
DataMap=pd.read_csv("dataset/ble_irrigue/MAR_Region/MBle_IRR_40.csv")
DataMap.tail(5)


# In[136]:



# Draw the complet graph for  
DataMap.plot(x='PA', y='P', style='x')
plt.title('Rendement 30')  
plt.xlabel('PA')  
plt.ylabel('Besoin Phosphore') 
plt.show()

#filtrage
DatMapm15=DataMap[DataMap['PA']<15]

# Draw first part of graph (linear equation)
DataMapm15.plot(x='PA', y='P', style='x')
plt.title('Rendement 20')  
plt.xlabel('PA')  
plt.ylabel('Besoin Phosphore') 
plt.show()



# # Multiple Linear Regression

# Graphe 1
X = DataMapm15[['PA']]
y = DataMapm15['P']


# # Training the Algorithm

# In[84]:


# Graph 1
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  


# Import `StandardScaler` from `sklearn.preprocessing`
# Graph 1 
from sklearn.preprocessing import StandardScaler

# Define the scaler 
scaler = StandardScaler().fit(X_train)

# Scale the train set
X_train = scaler.transform(X_train)

# Scale the test set
X_test = scaler.transform(X_test)



# And finally, to train the algorithm we execute the same code as before, using the fit() method of the LinearRegression class:


from sklearn.linear_model import LinearRegression,LogisticRegression
regressor = LinearRegression(fit_intercept=True)

#regressor=LogisticRegression(solver='newton-cg', multi_class='multinomial')
regressor.fit(X_train, y_train)


# As said earlier, in case of multivariable linear regression, the regression model has to find the most optimal coefficients for all the attributes. To see what coefficients our regression model has chosen, execute the following script:

# In[95]:


# Graph 1
print(" *********************** Fisrt Graph ******************************************* ")

coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  
coeff_df 


# In[97]:

#print(regressor.intercept_)  
#print(regressor.coef_)


# check that the coeffients are the expected ones.
print("Function for the first Graph") 
m = regressor.coef_[0]
b = regressor.intercept_
print(' y = {0} * x + {1}'.format(m, b))
print("Function for the second Graph") 
m1 = regressor1.coef_[0]
b1 = regressor1.intercept_
print(' y = {0} * x + {1}'.format(m1, b1))


# # Making Predictions

# In[35]:


# Graph 1
y_pred = regressor.predict(X_test)

# Graph 1 
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df  


# # Evaluating the Algorithm
# 

# In[38]:


from sklearn import metrics  
print("********************************** First Graph ***************************************************\n")
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  
print("\n ********************************** Second Graph ***************************************************")
print('Mean Absolute Error:', metrics.mean_absolute_error(y1_test, y1_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y1_test, y1_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y1_test, y1_pred)))  


Hi friends, you find here the complete code that I used : 

#from sklearn.metrics import accuracy_score
#score = accuracy_score(y_test,y_pred)
#print(score# -*- coding: utf-8 -*-

