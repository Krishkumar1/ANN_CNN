# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
num_features = 39
import numpy as np
import pandas as pd
import tensorflow as tf

dataset = pd.read_csv("Database_C_21_CNN.csv")
X = dataset.iloc[:3343, :num_features]
Y = dataset.iloc[:3343, num_features]




from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X.iloc[:, 2:])
X.iloc[:, 2:] = imputer.transform(X.iloc[:, 2:])
X.isnull().sum().sum()

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['HOME_TEAM_ID', 'VISITOR_TEAM_ID'])], remainder='passthrough')
X = np.array(ct.fit_transform(X))



from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units = 75, activation='relu'))

ann.add(tf.keras.layers.Dense(units = 75, activation='relu'))

ann.add(tf.keras.layers.Dense(units = 75, activation='relu'))
ann.add(tf.keras.layers.Dense(units = 75, activation='relu'))
ann.add(tf.keras.layers.Dense(units = 75, activation='relu'))






ann.add(tf.keras.layers.Dense(units = 1, activation='sigmoid'))


ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)


ann.fit(X_train, Y_train, batch_size = 32, epochs = 100)


y_pred = ann.predict(X_test)

y_pred = (y_pred > 0.5)



from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(Y_test, y_pred)

acc = accuracy_score(Y_test, y_pred)
print(acc)
