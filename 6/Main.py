#!/usr/bin/env python
# coding: utf-8

# In[167]:


import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

import keras.layers as layers
import keras.models as models
from keras.preprocessing.image import load_img, img_to_array
import cv2


# In[86]:


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)


# In[87]:


# train i test podaci
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


# In[88]:


# prikaz karakteristika train i test podataka
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))


# In[89]:


# TODO: prikazi nekoliko slika iz train skupa
n = 0
for i in x_train:
    plt.imshow(i)
    plt.show()
    n += 1
    if n > 10:
        break


# In[90]:


# skaliranje slike na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255


# In[91]:


# slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)


# In[92]:


print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0], "test samples")


# In[93]:


# pretvori labele
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)


# In[107]:


mc = keras.callbacks.ModelCheckpoint('./Model/',
                                     monitor="val_accuracy",
                                     save_best_only=True,
                                     verbose=1,
                                     mode="max",
                                     save_freq="epoch",
                                     options=None,)


# In[207]:


# TODO: kreiraj model pomocu keras.Sequential(); prikazi njegovu strukturu
model = models.Sequential()
model.add(layers.Input(shape=(28, 28, 1)))

# model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
# model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
# model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(layers.Dropout(rate=0.2))

# model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
# model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
# model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(layers.Dropout(rate=0.2))

model.add(layers.Flatten())
# model.add(layers.Dense(units=100, activation='relu'))
model.add(layers.Dense(units=10, activation='softmax'))

model.summary()


# In[205]:


# model = keras.models.load_model("./Model/")


# In[209]:


# TODO: definiraj karakteristike procesa ucenja pomocu .compile()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# TODO: provedi ucenje mreze
model.fit(x_train_s, y_train_s, epochs=1, validation_data=(x_test_s, y_test_s))


# In[210]:


# TODO: Prikazi test accuracy i matricu zabune
model.evaluate(x_test_s, y_test_s)


# In[211]:


# TODO: spremi model
model.save("./Model/")


# In[200]:


try:
    from timeit import default_timer as dt
    timer = True
except ImportError:
    timer = False

n = 0
for i in x_train:
    test = i[np.newaxis, :, :, np.newaxis]
    if timer:
        start = dt()
    # print(test.shape)
    guess = np.argmax(model.predict(test))
    if timer:
        print(str((dt() - start) * 1e3) + " ms")
    print(guess)
    
    plt.imshow(i)
    plt.show()
    n += 1
    if n > 10:
        break


# In[204]:


x = img_to_array(load_img(r"test.png", target_size=(28, 28)))[:,:,0]
x = cv2.dilate(x, kernel=(3, 3), iterations=1)

plt.imshow(x)
plt.show()

x = x[np.newaxis, :, :, np.newaxis]
# print(x.shape)

if timer:
    start = dt()
guess = np.argmax(model.predict(test))
if timer:
    print(str((dt() - start) * 1e3) + " ms")
print(guess)

