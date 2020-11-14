#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
#from IrisLocalization import IrisLocalization


# In[2]:



# transfer into a rectangle image 64*512
def getxy(X, Y, innerCircle, outterCircle):
    
    # getxy(X, Y, innerCircle, outterCircle):
    """transfer into a rectangle image 64*512
    :X,Y: coordinates
    :innerCircle, outterCircle: the center of radius of inner and outer circle
    :return: normalized coordiantes: (x,y)
    """
    
    # return the distance between two points
    def get_dist(x1,y1,x2,y2):
        return np.sqrt( (x1-x2)**2 + (y1-y2)**2 )
    
    # return the inverse of tangent between two points
    def get_inverseTan(x1,y1,x2,y2):
        tanTheta = (y2 - y1) / (x2 - x1)
        return np.arctan(tanTheta)
    
    # return the radius of the iris, which is longer than the radius of pupil
    def get_radius(d1, r2, theta):
        x1 = (2*d1*np.cos(theta) + np.sqrt( (2*d1*np.cos(theta))**2 - 4*(d1*d1 - r2*r2) )) / 2
        return x1

    
    (M, N) = (64, 512)
    theta = 2 * math.pi * X / N 
    (x_inner, y_inner, r_inner) = (innerCircle[0], innerCircle[1], innerCircle[2]) #inner circle info.
    (x_outer, y_outer, r_outer) = (outterCircle[0], outterCircle[1], outterCircle[2]) #outer circle info.
    d1 = get_dist(x_inner, y_inner, x_outer, y_outer) #the distance between 2 centers
    diffTheta = get_inverseTan(x_inner, y_inner, x_outer, y_outer) #the angle between 2 center vectors
    
    Radius = get_radius(d1, r_outer, diffTheta)
    
    x_inner = x_inner + r_inner * np.cos(theta)
    y_inner = y_inner + r_inner * np.sin(theta)
    
    x_outer = x_inner + Radius * np.cos(theta)
    y_outer = y_outer + Radius * np.sin(theta)
    
    x = int(x_inner + (x_outer - x_inner) * Y / M)
    y = int(y_inner + (y_outer - y_inner) * Y / M)
    
    x = min(319,x) or max(0,x)
    y = min(279,y) or max(0,y)
    
    return(x, y)
    


# For each pixel in normalized image, find the value for the corresponding pixels in the original image and fill in the value
    
def getNormalization(image, innerCircle, outterCircle):
    
    # getNormalization(image, innerCircle, outterCircle):
    """For each pixel in normalized image, find the value for the corresponding pixels in the original image and fill in the value
    :image: input image
    :innerCircle, outterCircle: the center of radius of inner and outer circle
    :return: normalized image: new_image
    """
    new_image = np.zeros((64,512))

    for Y in np.arange(64):
        for X in np.arange(512):
            (x, y) = getxy(X, Y, innerCircle, outterCircle)
            new_image[Y, 511- X] = image[y, x]
    return new_image


# In[3]:


# This function takes normalized image and rotate the rectangle image to specified degree

def getRotation(image, degree):
    # getRotation(image, degree):
    """This function takes normalized image and rotate the rectangle image to specified degree
    :image: input image
    :degree: rotate the rectangle image to specified degree
    :return: rotated image
    """
    pixels = abs(int(512*degree/360))
    if degree > 0:
        return np.hstack([image[:,pixels:],image[:,:pixels]] )
    else:
        return np.hstack([image[:,(512 - pixels):],image[:,:(512 - pixels)]] )


# In[ ]:




