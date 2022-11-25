import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("Data.csv")
x = data.iloc[:,:3].values
y = data.iloc[:,-1].values

# Missing Values
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(x[:,1:3])
x[:, 1:3] = imputer.transform(x[:,1:3])

#Encoding the indepent Variables
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

#Encoding the dependent Variables
label = LabelEncoder()
y = label.fit_transform(y)

print(x)
print(y)