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

y_predict = regressor.predict(x_test)

#Comparing the predicted value with the actual Value 