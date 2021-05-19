# -*- coding: utf-8 -*-
"""
Created on Thu May 13 06:33:13 2021

@author: karlygash.kussainova
"""


import pickle
import xlsxwriter
import openpyxl
from pathlib import Path
import numpy as np
import math
import statistics
import cv2
import PIL
from PIL import Image
import matplotlib 
from matplotlib import image
from matplotlib import pyplot
from numpy import asarray
import scipy
from scipy.stats import skew, kurtosis
from skimage import io
import skimage.measure    
import skimage.feature
import xlsxwriter
import pandas as pd
import seaborn as sns
# # How to use the saved model

with open ('classifier','rb') as f:
      mod= pickle.load(f)
    
filename_arg = "C:/Users/karlygash.kussainova/Downloads/Karlygash/Karlygash/MyFolder/1.png"    
#---------------------------------------------------------------
 # Read the file
 
def read_img_display(filename):
    im=Image.open(filename)
    
    #y_pred = mod.predict([[0.870655, 0.870655, -0.582299, 0.285894, 0.231365, -0.579999, -1.02436, 0.664007, 0.986286, 1.19648, -0.904533, -1.13354, 1.08772, 1.25481,-1.17386,-0.929994, 1.02572,1.10411,1.02446,0.61185,0.995131,-0.435454,-0.200913,-1.10796,-1.08801,0.838777]])
    
    #print (y_pred)
    
    image = asarray(im) 
    R = image[:,:,0]
    G = image[:,:,1]
    B = image[:,:,2] 
    
    # Red channel
    R = asarray(R) 
    flat = R.flatten()
    def get_histogram1(R, bins):
            histogram1 = np.zeros(bins)
            for pixel in R:
                histogram1[pixel] += 1
            return histogram1
    hist = get_histogram1(flat, 256)
        
    def cumsum(a1):
            a1 = iter(a1)
            b1 = [next(a1)]
            for i in a1:
                b1.append(b1[-1] + i)
            return np.array(b1)
    cs1 = cumsum(hist)
    nj1 = (cs1 - cs1.min()) * 255
    N1 = cs1.max() - cs1.min()
    cs1 = nj1 / N1
    cs1 = cs1.astype('uint8')
    img_new1 = cs1[flat]
    img_new1 = np.reshape(img_new1, R.shape)
    R = img_new1
    
     
    
    
    # Green channel
    G = asarray(G) 
    flat = G.flatten()
    def get_histogram1(G, bins):
            histogram1 = np.zeros(bins)
            for pixel in R:
                histogram1[pixel] += 1
            return histogram1
    hist = get_histogram1(flat, 256)
        
    def cumsum(a1):
            a1 = iter(a1)
            b1 = [next(a1)]
            for i in a1:
                b1.append(b1[-1] + i)
            return np.array(b1)
    cs1 = cumsum(hist)
    nj1 = (cs1 - cs1.min()) * 255
    N1 = cs1.max() - cs1.min()
    cs1 = nj1 / N1
    cs1 = cs1.astype('uint8')
    img_new1 = cs1[flat]
    img_new1 = np.reshape(img_new1, R.shape)
    G = img_new1
    
     
    
    # Blue channel
    B = asarray(B) 
    flat = R.flatten()
    def get_histogram1(B, bins):
            histogram1 = np.zeros(bins)
            for pixel in R:
                histogram1[pixel] += 1
            return histogram1
    hist = get_histogram1(flat, 256)
        
    def cumsum(a1):
            a1 = iter(a1)
            b1 = [next(a1)]
            for i in a1:
                b1.append(b1[-1] + i)
            return np.array(b1)
    cs1 = cumsum(hist)
    nj1 = (cs1 - cs1.min()) * 255
    N1 = cs1.max() - cs1.min()
    cs1 = nj1 / N1
    cs1 = cs1.astype('uint8')
    img_new1 = cs1[flat]
    img_new1 = np.reshape(img_new1, R.shape)
    B = img_new1
    
     
    
    #Feature Extraction
    Mean1=np.mean(R);
    Variance1=np.var(R)
    Variance1=math.sqrt(Variance1)
    Skewness1=skew(R.reshape(-1))
    Kurtosis1=kurtosis(R.reshape(-1))
    entropy1=skimage.measure.shannon_entropy(R)
    R=skimage.img_as_ubyte(R)
    g1=skimage.feature.greycomatrix(R,[1],[0],levels=256,symmetric=False,normed=True)
    Cont1=skimage.feature.greycoprops(g1,'contrast')[0][0]
    Energ1=skimage.feature.greycoprops(g1,'energy')[0][0]
    Homo1=skimage.feature.greycoprops(g1,'homogeneity')[0][0]
    Corre1=skimage.feature.greycoprops(g1,'correlation')[0][0]
    
     
    
    Mean2=np.mean(G);
    Variance2=np.var(G)
    Variance2=math.sqrt(Variance2)
    Skewness2=skew(G.reshape(-1))
    Kurtosis2=kurtosis(G.reshape(-1))
    entropy2=skimage.measure.shannon_entropy(G)
    G=skimage.img_as_ubyte(G)
    g2=skimage.feature.greycomatrix(G,[1],[0],levels=256,symmetric=False,normed=True)
    Cont2=skimage.feature.greycoprops(g2,'contrast')[0][0]
    Energ2=skimage.feature.greycoprops(g2,'energy')[0][0]
    Homo2=skimage.feature.greycoprops(g2,'homogeneity')[0][0]
    Corre2=skimage.feature.greycoprops(g2,'correlation')[0][0]
    
     
    
    Mean3=np.mean(B);
    Variance3=np.var(B)
    Variance3=math.sqrt(Variance3)
    Skewness3=skew(B.reshape(-1))
    Kurtosis3=kurtosis(B.reshape(-1))
    entropy3=skimage.measure.shannon_entropy(B)
    B=skimage.img_as_ubyte(B)
    g3=skimage.feature.greycomatrix(B,[1],[0],levels=256,symmetric=False,normed=True)
    Cont3=skimage.feature.greycoprops(g3,'contrast')[0][0]
    Energ3=skimage.feature.greycoprops(g3,'energy')[0][0]
    Homo3=skimage.feature.greycoprops(g3,'homogeneity')[0][0]
    Corre3=skimage.feature.greycoprops(g3,'correlation')[0][0]
    
    #--------------------------------------------------------------
    result = "Abnormal"
    y_pred= mod.predict( [[Mean1, Variance1, Skewness1, Kurtosis1,entropy1, Cont1,Energ1, Homo1, Corre1, Mean2, Variance2, Skewness2, Kurtosis2, entropy2, Cont2, Energ2, Homo2, Corre2, Mean3, Variance3, Skewness3, Kurtosis3, entropy3, Cont3, Energ3, Homo3, Corre3 ]])
    lst = y_pred.tolist()
    if lst[0] == 1:
        result = "Normal"
    else:
        result = "Abnormal"
        
    return (result)

#y_pred = mod.predict([[0.870655, 0.870655, -0.582299,0.285894,0.231365,-0.579999,-1.02436,0.664007,0.986286,	1.19648,-0.904533,-1.13354,	1.08772	1.25481,-1.17386,-0.929994,	1.02572,1.10411,1.02446,0.61185,0.995131,-0.435454,	-0.200913,-1.10796,	-1.08801,0.838777]])

read_img_display(filename_arg)