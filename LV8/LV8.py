#!/usr/bin/env python
# coding: utf-8

# In[2]:


import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os
import cv2
from tensorflow.keras.models import Sequential
from keras.layers import Flatten, Conv2D, MaxPooling2D, Dropout, Input, Dense
from keras.layers.experimental.preprocessing import Rescaling
import tensorflow.keras.layers
import matplotlib.pyplot as plt
import cv2
import h5py
from keras.preprocessing.image import load_img, img_to_array
import numpy as np


# In[3]:


def SortTest():
    temp = 0
    with open("./GTSRB/Test.csv") as f:
        for line in f.readlines():
            if temp == 0:
                temp += 1
                continue
            info = line.split(",")
            print(info)

            try:
                os.mkdir("./GTSRB/TestSorted/" + info[-2])
            except Exception:
                pass

            # print("./GTSRB/" + info[-1][:-1], info[-1][:-1].split("/")[-1])
            os.rename("./GTSRB/" + info[-1][:-1], "./GTSRB/TestSorted/" + info[-2] + "/" + info[-1][:-1].split("/")[-1])


# In[4]:


# SortTest()


# In[5]:


train_ds = image_dataset_from_directory(directory='GTSRB/Train/', labels='inferred', label_mode='categorical', batch_size=32, image_size=(48, 48))


# In[6]:


test_ds = image_dataset_from_directory(directory='GTSRB/TestSorted/', labels='inferred', label_mode='categorical', batch_size=32, image_size=(48, 48))


# In[7]:


trainGeneartor = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                              rotation_range=30,
                                                              width_shift_range=0.1,
                                                              height_shift_range=0.1,
                                                              zoom_range=0.2,
                                                              shear_range=0.15,
                                                              brightness_range=[0.75, 1.3],
                                                              # horizontal_flip=True,
                                                              # vertical_flip=True,
                                                              fill_mode='nearest')
testGenerator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)


# In[8]:


train_ds = trainGeneartor.flow_from_directory('GTSRB/Train', batch_size=32, target_size=(48, 48), class_mode='categorical')
test_ds = testGenerator.flow_from_directory('GTSRB/TestSorted', batch_size=32, target_size=(48, 48), class_mode='categorical')


# In[9]:


mc = keras.callbacks.ModelCheckpoint('./Model/',
                                     monitor="val_accuracy",
                                     save_weights_only=True,
                                     save_best_only=True,
                                     verbose=1,
                                     mode="max",
                                     save_freq="epoch",
                                     options=None,)


# In[18]:


def LoadModel():
    model = keras.models.load_model('./Model')
    model.input_shape
    model.evaluate(test_ds)

def TestImage():
    x = img_to_array(load_img(r"C:\Users\Davu\JupyterNotebook\LV8\GTSRB\TestSorted\15\00052.png", target_size=(48, 48)))
    print(x.shape)
    x = np.expand_dims(x, axis=0)
    print(x.shape)
    guess = model.predict(x)
    print(guess[0])
    print(np.argmax(guess[0]))


# In[19]:


model = Sequential()
model.add(Input(shape=(48, 48, 3, )))
# model.add(Rescaling(scale=1./255))

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.2))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.2))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.2))

model.add(Flatten())
# model.add(Dense(units=1000, activation='relu'))
model.add(Dense(units=43, activation='softmax'))

model.summary()


# In[20]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_ds, epochs=5, validation_data=test_ds, callbacks=[mc])


# In[57]:


model.save('./Model')

