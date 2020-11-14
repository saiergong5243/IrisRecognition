#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[2]:

# IrisLocalization(image)
"""localize pupil, using the method introduced in the paper.
:image: the input image
:return: return the center of radius of inner and outer circle, respectively: (innerCircle, outterCircle)
"""

def IrisLocalization(image):
    
    
    # get_minimum(image):
    """Roughly find the area where the center is located.
    :image: the input image
    :return: finding the row and column that has minimum sum of values: (xp, yp)
    """
    def get_minimum(image):
        image2 = image[60:240,100:220]
        #image2 = image[70:240, 70:280]
        ##horizontally
        h = np.sum(image2, 1) #each row
        #plt.plot(range(1, h.shape[0]+1), h)
        yp = np.argmin(h) + 60
    
    
        ##vertically
        v = np.sum(image2, 0) #each column
        #plt.plot(range(1, v.shape[0]+1), v)
        xp = np.argmin(v) + 100
    
        return (xp, yp)

    (xp, yp) = get_minimum(image)
    
    #choose the region 120*120 centered at (xp, yp)
    region120 = image[yp-60: yp+60, xp-60: xp+60]
    _, region = cv2.threshold(region120, 64, 65, cv2.THRESH_BINARY)
    
    #recalculate the center use the binary image
    region_binary = np.where(region>0, 1, 0)
    
    v = np.sum(region_binary, 0)
    min_x = np.argmin(v)
    h = np.sum(region_binary, 1)
    min_y = np.argmin(h)
    
    radius1 = (120 - np.sum(region_binary[min_y]))/2
    radius2 = (120 - np.sum(region_binary[:,min_x]))/2
    radius = int((radius1 + radius2)/2)
    
    xp = min_x + xp - 60
    yp = min_y + yp - 60
    
    
    region240 = image[np.arange(yp - 120, min(279, yp + 110)), :][:, np.arange(xp - 135, min(319, xp + 135))]
    region120 = image[np.arange(yp - 60, min(279, yp + 60)), :][:, np.arange(xp - 60, min(319, xp + 60))]
    
    #detect the circle for pupil
    for i in range(1, 5):
        circles_inner = cv2.HoughCircles(region120, cv2.HOUGH_GRADIENT, 1, 250, param1=50, param2=10,
                                   minRadius=(radius-i), maxRadius=(radius+i))
        if type(circles_inner) != type(None):
            break
        else:
            pass
    circles_inner = np.around(circles_inner)
    
    #detect the circle for iris
    circles_outer = cv2.HoughCircles(region240, cv2.HOUGH_GRADIENT, 1, 250, param1=30, param2=10,
                                     minRadius=98, maxRadius=118)
    circles_outer = np.around(circles_outer)
    
    
    
    # return the output and draw the boundary
    image1 = image.copy()

    for i in circles_inner[0,:]:
    # draw the inner circle
        cv2.circle(image1,(int(i[0]+ xp - 60), int(i[1] + yp - 60)), int(i[2]), (0,255,0), 2)
    # draw the center of the circle
        cv2.circle(image1,( int(i[0]+ xp - 60),int(i[1] + yp - 60)), int(i[2]), (0,255,0), 2)
        innerCircle = [i[0] + xp - 60, i[1] + yp -60, i[2]]


    for i in circles_outer[0,:]:
    # draw the outer circle
        cv2.circle(image1, (int(i[0]+ xp - 135), int(i[1] + yp - 120)), int(i[2]), (0, 255, 0), 2)
    # draw the center of the circle
        cv2.circle(image1, (int(i[0]+ xp - 135), int(i[1] + yp - 120)), int(i[2]), (0, 255, 0), 3)
        outterCircle = [int(i[0]+ xp - 135), int(i[1] + yp - 120), i[2]]
    
    #return the center of radius of inner and outer circle, respectively
    return(innerCircle, outterCircle)



#def IrisLocalization(image):
#    # Roughly find the area where the center is located    
#    def get_minimum(image):
#        #image2 = image[70:240,70:280] 
#        image2 = image[60:240, 100:220]
#        ##horizontally
#        h = np.sum(image2, 1) #each row
#        #plt.plot(range(1, h.shape[0]+1), h)
#        yp = np.argmin(h) + 60
#    
#    
#        ##vertically
#        v = np.sum(image2, 0) #each column
#        #plt.plot(range(1, v.shape[0]+1), v)
#        xp = np.argmin(v) + 100
#    
#        return (xp, yp)

#    (xp, yp) = get_minimum(image)
    
#    #choose the region 120*120 centered at (xp, yp)
#    region120 = image[yp-60: yp+60, xp-60: xp+60]
#    # Binarize this region with a threshold of 60              
#    _, region = cv2.threshold(region120, 60, 65, cv2.THRESH_BINARY)  
    
#    #recalculate the center use the binary image
#    region_binary = np.where(region>0, 1, 0)
    
#    v = np.sum(region_binary, 0)
#    min_x = np.argmin(v)
#    h = np.sum(region_binary, 1)
#    min_y = np.argmin(h)
    
#    radius1 = (120 - np.sum(region_binary[min_y]))/2
#    radius2 = (120 - np.sum(region_binary[:,min_x]))/2
#    radius = int((radius1 + radius2)/2)
    
#    xp = min_x + xp - 60
#    yp = min_y + yp - 60
    
    
#    region240 = image[np.arange(yp - 120, min(279, yp + 110)), :][:, np.arange(xp - 135, min(319, xp + 135))]
#    region120 = image[np.arange(yp - 60, min(279, yp + 60)), :][:, np.arange(xp - 60, min(319, xp + 60))]
    
    #detect the circle for pupil
#    for i in range(1, 5):
#        circles_inner = cv2.HoughCircles(region120, cv2.HOUGH_GRADIENT, 1, 250, param1=50, param2=10,
#                                   minRadius=(radius-i), maxRadius=(radius+i))
#        if type(circles_inner) != type(None):
#            break
#        else:
#            pass
#    circles_inner = np.around(circles_inner)
    
#    #detect the circle for iris
#    circles_outer = cv2.HoughCircles(region240, cv2.HOUGH_GRADIENT, 1, 250, param1=30, param2=10,
#                                     minRadius=98, maxRadius=118)
#    circles_outer = np.around(circles_outer)
    
    
    
#    # return the output and draw the boundary
#    image1 = image.copy()

#    for i in circles_inner[0,:]:
#    # draw the inner circle
#        cv2.circle(image1,(int(i[0]+ xp - 60), int(i[1] + yp - 60)), int(i[2]), (0,255,0), 2)
#    # draw the center of the circle
#        cv2.circle(image1,( int(i[0]+ xp - 60),int(i[1] + yp - 60)), int(i[2]), (0,255,0), 2)
#        innerCircle = [i[0] + xp - 60, i[1] + yp -60, i[2]]


#    for i in circles_outer[0,:]:
#    # draw the outer circle
#        cv2.circle(image1, (int(i[0]+ xp - 135), int(i[1] + yp - 120)), int(i[2]), (0, 255, 0), 2)
#    # draw the center of the circle
#        cv2.circle(image1, (int(i[0]+ xp - 135), int(i[1] + yp - 120)), int(i[2]), (0, 255, 0), 3)
#        outterCircle = [int(i[0]+ xp - 135), int(i[1] + yp - 120), i[2]]
    
    #return the center of radius of inner and outer circle, respectively
#    return(innerCircle, outterCircle)


# In[ ]:




