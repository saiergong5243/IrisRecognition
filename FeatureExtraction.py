#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

import scipy.signal


# In[2]:


# get the ROI :image[0:48,:]
  
# M1(x, y, sigma_y), G(x,y,sigma_x,sigma_y):
"""M1 and G are helper functions for defined filter
:x,y: x and y axis coordinates
:sigma_x,sigma_y: space constants of the Gaussian envelope along the x and y axis
:return: filter
"""    

# M1 and G are helper functions for defined filter
def M1(x, y, sigma_y):
    f = 1/sigma_y
    return np.cos(2 * math.pi *f * math.sqrt(x**2 + y**2) )
    
def G(x,y,sigma_x,sigma_y):
    return (1/(2 * math.pi * sigma_x * sigma_y) * math.exp( -(x**2/sigma_x**2 + y**2/sigma_y**2)/2) * M1(x,y,sigma_y) )
    

    
# get_kernel(sigma_x,sigma_y):
"""This function calculates the defined filter with specified siamgeX,sigmaY
:sigma_x,sigma_y: space constants of the Gaussian envelope along the x and y axis
:return: kernel
"""
# This function calculates the defined filter with specified siamgeX,sigmaY    
def get_kernel(sigma_x,sigma_y):
    kernel = np.zeros((9,9)) #use a 9*9 kernel
    for row in range(9):
        for col in range(9):
            kernel[row,col] = G( (-4+col),(-4+row),sigma_x,sigma_y)
    return kernel



# getFilteredImage(image,sigma_x,sigma_y):
"""This function calculates the convolution of image and filter
:sigma_x,sigma_y: space constants of the Gaussian envelope along the x and y axis
:return: filtered image
"""
# This function calculates the convolution of image and filter
def getFilteredImage(image,sigma_x,sigma_y):
    image = image[0:48,:]
    kernel = get_kernel(sigma_x,sigma_y)
    new_image = scipy.signal.convolve2d(image, kernel, mode='same')
    return new_image




# getFeatureVector(f1,f2):
"""This function takes the two convolved images and extracts mean and standard deviation for each 8*8 small block as the feature vector for a specific image
:f1,f2: two convolved images
:return: mean and standard deviation for each 8*8 small block as the feature vector
"""
# This function takes the two convolved images and extracts mean and standard deviation for each 8*8 small block as the feature vector for a specific image
def getFeatureVector(f1,f2):
    nrow = int(f1.shape[0]/8)
    ncol = int(f1.shape[1]/8)
    vector = np.zeros(nrow*ncol*2*2)
    for i in range(2):
        image = [f1,f2][i]
        for row in range(nrow):
            for col in range(ncol):
                meanValue = np.mean( np.abs( image[row*8: (row+1) * 8,col*8: (col +1) * 8] ))
                sdValue = np.sum(abs(image[row*8: (row+1) * 8,col*8: (col +1) * 8] - meanValue))/ (8*8)
                vector[i*768 + 2*row*ncol + 2*col] = meanValue
                vector[i*768 + 2*row*ncol + 2*col + 1] = sdValue
    return vector   


# In[ ]:




