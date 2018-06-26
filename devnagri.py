#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 13:03:56 2018

@author: rahul
"""

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import pickle as p
import numpy as np
import pandas as pd

#%%
batch_size = 128
epochs = 25

#%%
# the data, shuffled and split between train and test sets
devnagri_dataset = pd.read_csv('dataset/data.csv', sep=',')
#%%
devnagri_dataset_np = np.array(devnagri_dataset)
#%%
for i in range(0,4):
    np.random.shuffle(devnagri_dataset_np)
#%%
devnagri_dataset_X = devnagri_dataset_np[:,:-1]
devnagri_dataset_Y = devnagri_dataset_np[:,-1]
#%%
devnagri_dataset_X = devnagri_dataset_X.astype('float32')
devnagri_dataset_X /= 255

print(devnagri_dataset_np.shape[0], 'train samples')

#%%
#splitt dataset
from sklearn.cross_validation import train_test_split
x_train, x_test_valid, y_train, y_test_valid = train_test_split(devnagri_dataset_X, devnagri_dataset_Y, test_size=0.3, random_state=1)
x_test, x_valid, y_test, y_valid = train_test_split(x_test_valid, y_test_valid, test_size=0.5, random_state=1)
#%%
#Finding number of classes and equal distribution of classes
classes_name = np.unique(devnagri_dataset_Y)
num_classes_main = np.unique(devnagri_dataset_Y).shape[0]
num_classes_train = np.unique(y_train).shape[0]
num_classes_test = np.unique(y_test).shape[0]
num_classes_valid = np.unique(y_valid).shape[0]
#%%
print("num_classes_main: ", num_classes_main)
print("num_classes_train: ", num_classes_train)
print("num_classes_test: ", num_classes_test)
print("num_classes_valid: ", num_classes_valid)
#%%
dict={}
code=0
for class_n in classes_name:
    dict[class_n]=code
    code=code+1
#%%
for i in range(0,y_test.shape[0]):
    y_test[i] = dict[y_test[i]]
    
for i in range(0,y_train.shape[0]):
    y_train[i] = dict[y_train[i]]
    
for i in range(0,y_valid.shape[0]):
    y_valid[i] = dict[y_valid[i]]
    

#%%
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes_main)
y_test = keras.utils.to_categorical(y_test, num_classes_main)
y_valid = keras.utils.to_categorical(y_valid, num_classes_main)

#%%
#num_classes=51
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(1024,)))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes_main, activation='softmax'))
model.summary()

#%%
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
#%%
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

#%%
score = model.evaluate(x_test, y_test, verbose=0)
#%%
print('Test loss:', score[0]*100)
print('Test accuracy:', score[1]*100)

#%%