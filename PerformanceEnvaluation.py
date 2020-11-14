#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import metrics 

from IrisLocalization  import *
from IrisNormalization import *
from ImageEnhancement import *
from FeatureExtraction import *
from IrisMatching import *


# In[2]:




# getCRRCurve(train,test):
"""Recognition results using features of different dimentionality under Cosine Similarity Measure
:train, test: train and test features
:return: CRR curve
"""
def getCRRCurve(train,test):
    vec = []
    # dimention could also be changed into any integer between 1 and 107, I chose
    # these as samples 
    dimention = [50,60,70,80,90,100,107] #try different dimensions
    plt.figure()
    for i in range(len(dimention)):
        print('Currently computing dimention %d' %dimention[i])
        vec.append(getMatching(train,test,dimention[i])) #use the default distance--cosine
    lw = 2

    plt.plot(dimention, vec, color='darkorange',lw=lw)
    plt.xlabel('Dimensionality of the feature vector')
    plt.ylabel('Correct recgnition rate')
    plt.title('Recognition results using features of different dimentionality under Cosine Similarity Measure')
    plt.scatter(dimention,vec,marker='*')

    plt.show()


# getTable(train,test):
""" prints the accurate rate using different distance measures
:train, test: train and test features
:return: Feature vector
"""
# This function prints the accurate rate using different distance measures
def getTable(train,test):
    vec = []
    dimension = [100,107]
    for i in range(1,4):
        print('Currently computing distance measure number %d' %i)
        for dim in range(2):
            vec.append(getMatching(train,test,LDADimention=dimension[dim],distanceMeasure=i))
    vec = np.array(vec).reshape(3,2)
    vec = pd.DataFrame(vec)
    vec.index = ['L1 distance measure', 'L2 distance measure','Cosine similarity measure']
    vec.columns = ['Original Feature Set', 'Reduced Feature Set']
    print(vec)
    return vec



def ROC_curve(train, trainY, test, testY):
    accuracyRate80, prediction, score = getMatching2(train, test, LDADimention=80, distanceMeasure=3)
    result = []
    for i in range(len(prediction)):
        if prediction[i] == testY[i]:
            result.append(1)
        else:
            result.append(0)
    
    fpr,tpr,threshold = metrics.roc_curve(result, score)
    tpr = 1-tpr
    roc_auc = metrics.auc(tpr,fpr)
    
    plt.figure(figsize=(10,10))  
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)   
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  
    plt.xlim([0.0, 1.0])  
    plt.ylim([0.0, 1.05])  
    plt.xlabel('False Positive Rate')  
    plt.ylabel('True Positive Rate')  
    plt.title('ROC under n_components = 80 and Cosine Similarity')  
    plt.legend(loc="lower right")  
    plt.show()  

# In[ ]:




