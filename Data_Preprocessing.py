import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("Data.csv")
x = data.iloc[:,:3].values
y = data.iloc[:,-1].values

# Missing Values
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(x[:,1:3])
x[:, 1:3] = imputer.transform(x[:,1:3])

#Encoding the independent Variables
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

#Encoding the dependent Variables
label = LabelEncoder()
y = label.fit_transform(y)

#Spliting the training data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=1)

print(x_train)
print("----------")
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

print(x_train)
print("-----------")
print(x_test)
