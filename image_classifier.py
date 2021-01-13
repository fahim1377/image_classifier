#!/usr/bin/env python
# coding: utf-8

# Imports

# In[1]:


import cv2
from keras.datasets import cifar10
import os 
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from random import randrange
import matplotlib.pyplot as plt

# Read data

# In[2]:


def load_data():
    #read train data
    train_dir = os.path.abspath('dataset/train/images/')             #directory which contains training images
    x_train = []
    y_train = []
    for label, cls in enumerate(os.listdir(train_dir)):               # there is different classes directories in train_dir
        print(cls)
        for image_path in os.listdir(train_dir+'/'+cls):
            img = cv2.imread(train_dir+'/'+cls+'/'+image_path)
            # downsample image to 32x32x3 using interpolation
            img = cv2.resize(img, dsize=(32,32), interpolation = cv2.INTER_AREA) 
            x_train.append(img)
            y_train.append(label)
            
    x_train = np.array(x_train,dtype='float')
    y_train = np.array(y_train)
    
    #read test data
    test_dir = os.path.abspath('dataset/test/images/')             #directory which contains training images
    x_test = []
    y_test = []
    for label, cls in enumerate(os.listdir(test_dir)):               # there is different classes directories in train_dir
        for image_path in os.listdir(test_dir+'/'+cls):
            img = cv2.imread(test_dir+'/'+cls+'/'+image_path)
            # downsample image to 32x32x3 using interpolation
            img = cv2.resize(img, dsize=(32,32), interpolation = cv2.INTER_AREA) 
            x_test.append(img)
            y_test.append(label)
    x_test = np.array(x_test,dtype='float')
    y_test = np.array(y_test)
    
    return (x_train, y_train), (x_test, y_test), len(os.listdir(test_dir))
    


# In[ ]:


(x_train, y_train), (x_test, y_test), n_classes = load_data()

# Preproccesing

# In[4]:


def preproccessing(x_train, x_test, y_train, y_test, n_classes):
    #normalize data
    x_train /= 255
    x_test /= 255
    
    #convert label to one hot key model
    y_train = np_utils.to_categorical(y_train, n_classes)
    y_test = np_utils.to_categorical(y_test, n_classes)
    return x_train, x_test, y_train, y_test


# In[5]:


x_train, x_test, y_train, y_test = preproccessing(x_train, x_test, y_train, y_test, n_classes)


# In[6]:


def rotate(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


# In[7]:


def data_augmentation(x_train, y_train):
    augmentaded_x = []
    augmentaded_y = []
    for count,img in enumerate(x_train):
        augmentaded_x.append(img)
        augmentaded_y.append(y_train[count])
        
        angle = randrange(180)
        augmentaded_x.append(rotate(img, angle, (16,16)))
        augmentaded_y.append(y_train[count])
        
        angle = randrange(180)
        augmentaded_x.append(rotate(img, angle, (16,16)))
        augmentaded_y.append(y_train[count])
        
        angle = randrange(180)
        augmentaded_x.append(rotate(img, angle, (16,16)))
        augmentaded_y.append(y_train[count])
        
        angle = randrange(180)
        augmentaded_x.append(rotate(img, angle, (16,16)))
        augmentaded_y.append(y_train[count])
        
        
    augmentaded_x = np.array(augmentaded_x,dtype='float')
    augmentaded_y = np.array(augmentaded_y,dtype='float')
    
    return augmentaded_x,augmentaded_y

x_train, y_train = data_augmentation(x_train, y_train)
t= x_train[1]
plt.imshow(cv2.cvtColor(np.float32(t), cv2.COLOR_BGR2RGB))
plt.show()


# create model

# In[8]:


model = Sequential()

# convolutional layer
model.add(Conv2D(50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(32, 32, 3)))

# convolutional layer
model.add(Conv2D(75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(125, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# flatten output of conv
model.add(Flatten())

# hidden layer
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.3))
# output layer
model.add(Dense(19, activation='softmax'))

# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# training the model for 10 epochs
model.fit(x_train, y_train, batch_size=128, epochs=1, validation_data=(x_test, y_test))


# In[4]:



# In[10]:


# In[11]:





