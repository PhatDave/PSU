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


# In[5]:


dt.ConfigurePlt([8])


# In[6]:


plt.scatter(data[:,0], data[:,1])


# In[7]:


clusters = 3
kmeans = cluster.KMeans(n_clusters=clusters, n_init=1)
kmeans.fit(data) 
values = kmeans.cluster_centers_.squeeze()
labels = kmeans.labels_

print(labels.shape, data.shape)


# In[8]:


formatData = [[data[i], labels[i]] for i in range(data.shape[0])]
formatData.sort(key=lambda x: x[1])


# In[9]:


sortData = []
for i in range(clusters):
    temp = []
    for j in formatData:
        if j[1] != i:
            continue
        else:
            temp.append([j[0][0], j[0][1]])
            #temp.append(j[0])
    sortData.append(temp)
sortData = np.array(sortData)


# In[10]:


sortDataAgain = []
for j in range(clusters):
    sortDataX1 = [sortData[j][i][0] for i in range(len(sortData[j]))]
    sortDataX2 = [sortData[j][i][1] for i in range(len(sortData[j]))]
    sortDataAgain.append([sortDataX1, sortDataX2])
    
colors=["#0000FF", "#00FF00", "#FF0066"]


# In[13]:


c = 0
for i in sortDataAgain:
    plt.scatter(i[0], i[1], color=colors[c])
    c += 1

plt.scatter(values[:,0], values[:,1], color='black')


# In[ ]:




