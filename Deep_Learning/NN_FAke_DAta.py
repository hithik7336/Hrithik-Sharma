# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Importing The Libraries

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

# Getting The Whole Dataset

pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',1000)
pd.set_option('display.max_colwidth',1000)
pd.set_option('display.width',1000)

# Getting The Dataset

data = pd.read_csv(r'D:\original\TensorFlow_FILES\DATA\fake_reg.csv')

# Checking On The Data

print(data.shape)

print(data.columns)

print(data.index)

print(data.info())

print(data.describe())

# Doing Some Exploratory Data Analysis

sns.scatterplot(x = 'feature1', y = 'feature2', data = data, palette = 'autumn')
plt.show()

sns.pairplot(data = data, palette = 'autumn')
plt.show()

# Dividing Dataset In Dependent & Independent Variables

X = data[['feature1', 'feature2']].values
y = data['price'].values

# Diving Data In Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# Scaling The Data

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)

# Creating The Neural Network

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(4,activation = 'relu'))
model.add(Dense(4,activation = 'relu'))
model.add(Dense(4,activation = 'relu'))

model.add(Dense(1))

model.compile(optimizer = 'rmsprop', loss = 'mse')

# Training The Neural Network

model.fit(x = X_train, y = y_train, epochs = 250, verbose = 1)

# Evaluation Of Model

model.evaluate(X_test, y_test, verbose = 1)

model.evaluate(X_train, y_train, verbose = 1)

# Predictions

y_pred = model.Predict(X_test)













