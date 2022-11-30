import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Importing The dataset
dataset = pd.read_csv('Salary_Data.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# print(x)
# print(y)

#Spliting the data

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

#Training a simple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Predicting the Test set

y_pred = regressor.predict(x_test)

# plt.scatter(x_train, y_train, color='red')
# plt.plot(x_train, regressor.predict(x_train), color='blue')
# plt.title('Salary vs Experience (Training Dataset)')
# plt.xlabel('Years of experience')
# plt.ylabel('Salary')
# plt.show()

#Testing the test dataset
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Testing Dataset)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

print(regressor.predict([[10]]))
# print(regressor.coef_)
# print(regressor.intercept_)