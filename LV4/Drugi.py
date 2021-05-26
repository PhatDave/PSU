#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
import numpy as np
import daveTrash as dt
import seaborn as sns


# In[2]:


def generate_data(n_samples, flagc):
    
    if flagc == 1:
        random_state = 365
        X,y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
        
    elif flagc == 2:
        random_state = 148
        X,y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)
        
    elif flagc == 3:
        random_state = 148
        X, y = datasets.make_blobs(n_samples=n_samples,
                                    centers=4,
                                    cluster_std=[1.0, 2.5, 0.5, 3.0],
                                    random_state=random_state)

    elif flagc == 4:
        X, y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
        
    elif flagc == 5:
        X, y = datasets.make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X


# In[3]:


import scipy as sp
from sklearn import cluster, datasets
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


data = generate_data(500, True)
print(data.shape)


# In[6]:


dt.ConfigurePlt([8])


# In[7]:


plt.scatter(data[:,0], data[:,1])


# In[30]:


kmeans = cluster.KMeans(n_clusters=3, n_init=1)
kmeans.fit(data) 
values = kmeans.cluster_centers_.squeeze()
labels = kmeans.labels_

print(labels.shape, data.shape)


# In[29]:


print(labels, values)
plt.scatter(values[:,0], values[:,1], color='red')
plt.scatter(data[:,0], data[:,1], color='green')


# In[ ]:




