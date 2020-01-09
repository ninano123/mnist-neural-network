#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.special


# In[2]:


data_file = open("mnist_dataset/mnist_train_100.csv",'r')
data_list = data_file.readlines()
data_file.close()


# In[3]:


len(data_list)


# In[4]:


data_list[0]


# In[5]:


all_values = data_list[0].split(',')
image_array = np.asfarray(all_values[1:]).reshape((28,28))
plt.imshow(image_array, cmap='Greys', interpolation='None')


# In[6]:


scaled_input = (np.asfarray(all_values[1:]) /255.0 * 0.99) +0.01
print(scaled_input)


# In[7]:


output_nodes = 10
targets = np.zeros(output_nodes) +0.01
targets[int(all_values[0])] =0.99
print(targets)


# In[ ]:




