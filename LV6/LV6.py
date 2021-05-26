#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix
import daveTrash as dt
import matplotlib.pyplot as plt
import joblib
import pickle
import numpy as np
import matplotlib.image as mpimg
from skimage import color
from skimage.transform import resize


# In[ ]:


# Load additional data


# In[3]:


X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)


# In[23]:


dt.ConfigurePlt(mode=False, size=9, dpi=100)


# In[5]:


def ShowSet(data, start=0, N=5):
    show = []
    show2 = []

    for i in range(start, start + N):
        show.append(data[i].reshape(28, 28))
    for i in range(start + N, start + (N * 2)):
        show2.append(data[i].reshape(28, 28))

    plt.imshow(np.vstack((np.hstack(show), np.hstack(show2))))


# In[6]:


ShowSet(X, 2397, 5)


# In[7]:


# skaliraj podatke, train/test split
X = X / 255.
XTrain, XTest = X[:60000], X[60000:]
YTrain, YTest = y[:60000], y[60000:]


# In[22]:


# TODO: izgradite vlastitu mrezu pomocu sckitlearn MPLClassifier 
bogec = MLPClassifier(verbose=1, max_iter=100, hidden_layer_sizes=(100, 20, 20, ))
#bogec.n_layers_int = 5
bogec.fit(XTrain, YTrain)


# In[24]:


# TODO: evaluirajte izgradenu mrezu
results = bogec.predict(XTest)
start = 100
print(results[start:start + 10])
print(YTest[start:start + 10])
ShowSet(XTest, start=100, N=5)


# In[25]:


good = np.where(results == YTest, 1, 0)
print(sum(good), len(YTest), (sum(good) / len(YTest)) * 100)


# In[26]:


# spremi mrezu na disk
filename = "bogec.sav"
joblib.dump(bogec, filename)


# In[12]:


dt.ConfigurePlt(size=3)


# In[13]:


# ucitaj sliku i prikazi ju
filename = 'test.png'

img = mpimg.imread(filename)
img = color.rgb2gray(img)
img = resize(img, (28, 28))

plt.figure()
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()

# TODO: prebacite sliku u vektor odgovarajuce velicine
img = img.reshape(1, 28 * 28)

# vrijednosti piksela kao float32
img = img.astype('float32')


# In[14]:


bogec.predict(img.reshape(1, 28 * 28))


# In[ ]:




