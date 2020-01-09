#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


a = np.zeros([3,2])
print(a)
type(a)


# In[4]:


a[0] = [1,2]
a[1] = [9,0]
a[2] = [0,12]
print(a)
type(a)


# In[7]:


plt.imshow(a, interpolation="nearest")


# In[ ]:





# In[ ]:




