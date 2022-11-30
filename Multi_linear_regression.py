import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Data Preprocessing
data = pd.read_csv("50_Startups.csv")
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

#One hot Encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

#Spliting the data set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)

#Training the Data

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(x_train,y_train)

#Comparing the predicted value with the actual Value
# np.set_printoptions(2)

y_predict = regressor.predict(x_test)

concatenate = np.concatenate((y_predict.reshape(len(y_predict),1), y_test.reshape(len(y_test),1)), 1)
print(concatenate)

#Printing individual
# print(x_test)
# print(regressor.predict([[0.0, 1.0, 0.0, 66051.52, 182645.56, 118148.2]]))

# Printing Coefficient
# print(regressor.coef_)
# print(regressor.intercept_)