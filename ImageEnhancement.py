#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


# In[61]:


#image = cv2.imread('try.bmp', 0)
#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# In[2]:

# enhancement(image):
"""enhance the image by cv2.equalizeHist
:image: input image
:return: enhanced image
"""

# enhance the image by cv2.equalizeHist
def enhancement(image):
    image = np.array(image,dtype=np.uint8) #reset the data type to use in cv2.equalizeHist()
    image = cv2.equalizeHist(image)
    return image


# imageEnhancement32by32(image): 
"""enhance image by histogram equalization by 32*32 pixels region
:image: input image
:return: enhanced image
"""

# enhance image by histogram equalization by 32*32 pixels region
def imageEnhancement32by32(image):                         
    nrow = int(image.shape[0]/32)
    ncol = int(image.shape[1]/32)
    for row in range(nrow):
        for col in range(ncol):
            enhanced = enhancement(image[row*32:(row+1)*32,col*32:(col+1)*32])
            image[row*32:(row+1)*32,col*32:(col+1)*32] = enhanced
    return image


# Enhancement(image):   
"""Sum up function for Image Enhancement part
:image: input image
:return: enhanced image
"""

# Sum up function for Image Enhancement part
def Enhancement(image):                               
    #image = subtractBackground(image)
    image = imageEnhancement32by32(image)
    return image


# In[ ]:




