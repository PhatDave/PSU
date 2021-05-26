#!/usr/bin/env python
# coding: utf-8

# In[99]:


from sklearn import datasets
import numpy as np
import daveTrash as dt
import seaborn as sns
import scipy as sp
from sklearn import cluster, datasets
import numpy as np
import matplotlib.pyplot as plt
import cv2


# In[141]:


img = cv2.imread("example_grayscale.png")
#img = img[:,:,::-1]


# In[142]:


plt.imshow(img)
face = img
print(img.shape)


# In[144]:


X = face.reshape((-1, 1)) # We need an (n_sample, n_feature) array
k_means = cluster.KMeans(n_clusters=2,n_init=1)
k_means.fit(X) 
values = k_means.cluster_centers_.squeeze()
labels = k_means.labels_
face_compressed = np.choose(labels, values)
face_compressed.shape = face.shape


# In[145]:


face_compressed /= 255


# In[146]:


plt.figure(1)
plt.imshow(face,  cmap='gray')
plt.show()

plt.figure(2)
plt.imshow(face_compressed,  cmap='gray')
plt.show()


# In[ ]:




