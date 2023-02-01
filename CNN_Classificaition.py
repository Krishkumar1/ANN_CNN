# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
num_features = 39
import numpy as np
import pandas as pd
from PIL import Image as im
import tensorflow as tf
val = 46
# j = 3343
j = 1964
dataset = pd.read_csv("Database_C_21_CNN.csv")
X = dataset.iloc[:j, :num_features]
Y = dataset.iloc[:j, num_features]

X = X.drop(['Days_from_21_March_2021'], axis = 1)


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

Y_test = Y_test.reset_index(drop=True)
Y_train = Y_train.reset_index(drop=True)

X_train_H = X_train[:, [0,	1,	2,	3,	4,	5,	6,	7,	8,	9, 10, 11,	12,	13,	14,	15,	16,	17,	18,	19, 20,	21,	22,	23,	
                        24,	25,	26,	27,	28,	29, 60,	61,	62,	63,	64,	65,	66,	67,	68,	69, 70,	71,	72,	73,	74,	75,	76,	77]]

X_train_A = X_train[:, [30,	31,	32,	33,	34,	35,	36,	37,	38,	39, 40,	41,	42,	43,	44,	45,	46, 47,	48,	49, 50,	51,	52,	53,	
                        54,	55,	56,	57,	58,	59, 60, 78,	79,80,	81,	82,	83,	84,	85,	86,	87,	88,	89,90,	91,	92,	93,	94,	95]]



b = 0
for b in range(1571):
    i = 0

    a = np.zeros(shape=(val, val), dtype= float)
    for i in range(val):
        j = 0
        for j in range(val):
            # if X_train_H[b][j] > X_train_A[b][i]:
            #     a[i][j] = 1
            # elif X_train_H[b][j] < X_train_A[b][i]:
            #     a[i][j] = -1
            # else:
            #     a[i][j] = 0
            a[i][j] = X_train_H[b][j] - X_train_A[b][i]

                   
            
    a = np.reshape(a, (val, val))
    data = im.fromarray(a)
    data = data.convert("RGB")
    name = ""
    name = "images/train/" + str(Y_train[b]) + "/image_" + str(b) + ".png"
    data.save(name)
    
      
X_test_H = X_train[:, [0,	1,	2,	3,	4,	5,	6,	7,	8,	9, 10, 11,	12,	13,	14,	15,	16,	17,	18,	19, 20,	21,	22,	23,	
                        24,	25,	26,	27,	28,	29, 60,	61,	62,	63,	64,	65,	66,	67,	68,	69, 70,	71,	72,	73,	74,	75,	76,	77]]

X_test_A = X_train[:, [30,	31,	32,	33,	34,	35,	36,	37,	38,	39, 40,	41,	42,	43,	44,	45,	46,	47,	48,	49, 50,	51,	52,	53,	
                        54,	55,	56,	57,	58,	59, 60, 78,	79, 80,	81,	82,	83,	84,	85,	86,	87,	88,	89,90,	91,	92,	93,	94,	95]]

Y_test = Y_test.reset_index(drop=True)

g = 0
for g in range(392):
    i = 0
    
    a = np.zeros(shape=(val, val), dtype=np.uint8)
    for i in range(val):
        j = 0
        for j in range(val):
            if X_test_H[b][j] > X_test_A[b][i]:
                a[i][j] = 1
            elif X_test_H[b][j] < X_test_A[b][i]:
                a[i][j] = -1
            else:
                a[i][j] = 0
    a = np.reshape(a, (val, val))
    data = im.fromarray(a)
    data = data.convert("RGB")
    name = ""
    name = "images/test/" + str(Y_test[g]) + "/image_" + str(g) + ".png"
    data.save(name)
    


from keras.preprocessing.image import ImageDataGenerator 

training_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip= True
    )     

train_set = training_datagen.flow_from_directory(
    'images/train',
    target_size=(val,val),
    batch_size=32,
    class_mode='binary'
    )

test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
    'images/test',
    target_size=(val, val),
    batch_size=32,
    class_mode='binary'
    )

cnn = tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(
        activation = 'relu',
        filters = 50,
        kernel_size = 3,
        input_shape = (val, val, 3)
    ))

cnn.add(tf.keras.layers.MaxPool2D(
    pool_size = 2,
    strides = 2
    ))



cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(
    units = 25,
    activation = 'relu'
    ))

cnn.add(tf.keras.layers.Dense(
    units = 25,
    activation = 'relu'
    ))


cnn.add(tf.keras.layers.Dense(
    units = 1,
    activation = 'sigmoid'
    ))

cnn.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
    )
                
cnn.fit(
        x = train_set,
        validation_data = test_set,
        epochs = 30
        )                     
                    
                
                          
                
                
                          
                    
                
                          
                