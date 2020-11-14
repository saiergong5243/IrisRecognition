#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import cv2
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


from IrisLocalization  import *
from IrisNormalization import *
from ImageEnhancement import *
from FeatureExtraction import *


# In[4]:


# Initially,  for each fileName, read the image, then do Iris Localization, IrisNormalization, ImageEnhancement, and FeatureExtraction.


# processImage(fileName):
"""This function is used for both training images and testing images
:filename: input image
:return: Feature vector
"""

#This function is used for both training images and testing images
def processImage(fileName):
    img = cv2.imread(fileName)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (innerCircle,outterCircle) = IrisLocalization(image)
    image = getNormalization(image,innerCircle,outterCircle)
    image = Enhancement(image)
    (image1,image2) = (getFilteredImage(image, sigma_x=3, sigma_y=1.5), getFilteredImage(image, sigma_x=4.5, sigma_y=1.5) )
    vector = getFeatureVector(image1, image2)
    return vector



# processImageWithRotation(fileName, degree):
"""This function only applies to training images, since we need do rotate the images with different angles
:filename: input image
:degree: rotation degree
:return: Feature vector
"""

#This function only applies to training images, since we need do rotate the images with different angles
def processImageWithRotation(fileName, degree):
    img = cv2.imread(fileName)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (innerCircle,outterCircle) = IrisLocalization(image)
    image = getNormalization(image, innerCircle, outterCircle)
    image = getRotation(image, degree)
    image = Enhancement(image)
    (image1, image2) = (getFilteredImage(image, sigma_x=3, sigma_y=1.5), getFilteredImage(image, sigma_x=4.5, sigma_y=1.5) )
    vector = getFeatureVector(image1, image2)
    return vector


# getDatabase(folder):
"""This function read all the training images and testing images respectively and return as a vector of features for each image.
:folder: folder directory
:return: list contains training image features or testing image features
"""
#This function read all the training images and testing images respectively and return as a vector of features for each image.
def getDatabase(folder):
    number = folder + 2
    folder = str(folder)
    vec = []
    # Folder 1 contains training image, we need to do with rotations
    if folder =='1':
        rotation = [-9, -6, -3, 0, 3, 6, 9] #seven angles
        
        for i in range(1, 109):
            for j in range(1, number+1):
                thisFileName = './CASIA Iris Image Database (version 1.0)/'
                index = "%03d" % (i,)
                fileName = thisFileName + index + '/'+folder+'/' + index +'_'+folder+'_%d' %(j) +'.bmp'
                print(fileName)
                for q in range(7):
                    vec.append(processImageWithRotation(fileName, rotation[q]/3))
                    
    # Folder 2 contains testing images, rotation is not needed.
    else:
        for i in range(1, 109):
            for j in range(1, number+1):
                thisFileName = './CASIA Iris Image Database (version 1.0)/'
                index = "%03d" % (i,)
                fileName = thisFileName + index + '/'+folder+'/' + index +'_'+folder+'_%d' %(j) +'.bmp'
                print(fileName)
                vec.append(processImage(fileName))
    return vec #list contains training image features or testing image features


# getMatching(train, test, LDADimention=107, distanceMeasure=3):
"""return the accuracy under some dimension and some distance measurement.
:train, test: train and test features
:return: accuracy
"""

def getMatching(train, test, LDADimention=107, distanceMeasure=3):
    #we need to change the train and test features from list to array.
    trainX = np.array(train)
    testX  = np.array(test)
    irisY = np.arange(1,109) # training labels
    trainY = np.repeat(irisY,3*7)
    testY = np.repeat(irisY,4)
    trainClass = np.repeat(irisY,3) #testing true labels
    
    clf = LDA(n_components = LDADimention) #do the LDA algorithm with default n_components which can be modified.
    clf.fit(trainX,trainY) #fit the model on the training dataset
    newTrain = clf.transform(trainX) #get the reduced training features
    newTest = clf.transform(testX) # get the reduced testing features
    
    
    
    #calculating the accuracy under certain n_component and certain distance measurement.
    predicted = np.zeros(testX.shape[0])
    for i in range(testX.shape[0]):
        vec = np.zeros(int(trainX.shape[0]/7))
        thisTest = newTest[i:i+1]
        for j in range(len(vec)):
            distance = np.zeros(7)
            for q in range(7):
                if distanceMeasure ==3:
                    distance[q] = scipy.spatial.distance.cosine(thisTest,newTrain[j*7+q:j*7+q+1]) #cosine distance
                elif distanceMeasure ==1:
                    distance[q] = scipy.spatial.distance.cityblock(thisTest,newTrain[j*7+q:j*7+q+1]) #L1
                else:
                    distance[q] = scipy.spatial.distance.sqeuclidean(thisTest,newTrain[j*7+q:j*7+q+1]) #L2 
                
            vec[j] = np.min(distance)
        shortestDistanceIndex = np.argmin(vec)
        predicted[i] = trainClass[shortestDistanceIndex]
    
    predicted = np.array(predicted,dtype =np.int) #the predicted labels for each testing image
    accuracyRate = 1 - sum(predicted != testY)/len(testY)
    return accuracyRate #return the accuracy under some dimension and some distance measurement.


# In[ ]:


def getMatching2(train, test, LDADimention=107, distanceMeasure=3):
    #we need to change the train and test features from list to array.
    trainX = np.array(train)
    testX  = np.array(test)
    irisY = np.arange(1,109) # training labels
    trainY = np.repeat(irisY,3*7)
    testY = np.repeat(irisY,4)
    trainClass = np.repeat(irisY,3) #testing true labels
    
    clf = LDA(n_components = LDADimention) #do the LDA algorithm with default n_components which can be modified.
    clf.fit(trainX,trainY) #fit the model on the training dataset
    newTrain = clf.transform(trainX) #get the reduced training features
    newTest = clf.transform(testX) # get the reduced testing features
    
    
    
    #calculating the accuracy under certain n_component and certain distance measurement.
    predicted = np.zeros(testX.shape[0])
    d = np.zeros(testX.shape[0])
    for i in range(testX.shape[0]):
        vec = np.zeros(int(trainX.shape[0]/7))
        thisTest = newTest[i:i+1]
        for j in range(len(vec)):
            distance = np.zeros(7)
            for q in range(7):
                if distanceMeasure ==3:
                    distance[q] = scipy.spatial.distance.cosine(thisTest,newTrain[j*7+q:j*7+q+1]) #cosine distance
                elif distanceMeasure ==1:
                    distance[q] = scipy.spatial.distance.cityblock(thisTest,newTrain[j*7+q:j*7+q+1]) #L1
                else:
                    distance[q] = scipy.spatial.distance.sqeuclidean(thisTest,newTrain[j*7+q:j*7+q+1]) #L2 
                
            vec[j] = np.min(distance)
        shortestDistanceIndex = np.argmin(vec)
        d[i] = vec[shortestDistanceIndex]
        predicted[i] = trainClass[shortestDistanceIndex]
    
    predicted = np.array(predicted,dtype =np.int) #the predicted labels for each testing image
    accuracyRate = 1 - sum(predicted != testY)/len(testY)
    return accuracyRate, predicted, d #return the prediction and shorest distance of each testing image.




