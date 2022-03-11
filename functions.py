#!/usr/bin/env python3

#ENPM673 Spring 2022
#Section 0101
#Jerry Pittman, Jr. UID: 117707120
#jpittma1@umd.edu
#Project #2 Functions

import numpy as np
import cv2
import scipy
from scipy import fft, ifft
from numpy import linalg as LA
import matplotlib.pyplot as plt
import sys
import math
import os
from os.path import isfile, join

'''Problem 1 Functions'''
def convertImagesToMovie(folder):
    fps=3
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoname=('night_drive')
    frames = []
    
    files = [f for f in os.listdir(folder) if isfile(join(folder, f))]
    
    #pictures were not in order, sadly
    files.sort(key = lambda x: x[5:-4])
    files.sort()
    
    for i in range(len(files)):
        filename=folder + files[i]
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        # cv2.imshow('image', img)
        # print("size is ", size)
        if img is not None:
            frames.append(img)
        else:
            print("Failed to read")
    
    video = cv2.VideoWriter(str(videoname)+".avi",fourcc, fps, size)
    
    for i in range(len(frames)):
        # writing to a image array
        video.write(frames[i])
    
    # video.release()
    
    return video

def createHistogramFigure(img):
    # a = np.zeros(256, np.int32)
     
    # height,width, _ = img.shape
     
    # for i in range(width):
    #     for j in range(height):
    #         g = img[j,i]
    #         a[g] = a[g]+1
            
    # # print("hist is ", a)
    
    plt.hist(a,bins=255)
    # plt.show()
    plt.savefig('histogram_frame2.png')

# def conductHistogramEqualization(img,bins):
#     # bins = np.zeros((256,),dtype=np.float16)
#     a = bins
#     b = bins
#     # a = np.zeros((256,),dtype=np.float16)
#     # b = np.zeros((256,),dtype=np.float16)
             
#     height,width, _ = img.shape

#     #Create histogram
#     for i in range(width):
#         for j in range(height):
#             g = img[j,i]
#             a[g] = a[g]+1

#     # print("hist is ", a)
    
#     '''Cumulative Distibution Function
#     Number of pixels with intensity less than or equal to "i" intensity
#     normalized by N pixels'''
#     #performing histogram equalization
#     N = 1.0/(height*width)

#     for i in range(256):
#         for j in range(i+1):
#             b[i] += a[j] * N;
#             # print("b[i] before", b[i])
#         b[i] = round(b[i] * 255);
#         # print("b[i] after", b[i])
        

#     # # cum_dist_funct=b
#     cum_dist_funct=b.astype(np.uint8)

#     # # print("CDF is ", cum_dist_funct)

#     #Equalize Image based on CDF results
#     for i in range(width):
#         for j in range(height):
#             g = img[j,i]
#             img[j,i]= cum_dist_funct[g]
     
#     # cv2.imshow('image',img)
#     # cv2.waitKey(0)
    
#     return img
    

def createHistogram(input):
    hist_vals = list()
    
    for i in range(256):
        c = np.where(input == i)
        hist_vals.append( [ i , len(c[0]) ] )
        
    return hist_vals


def conductHistogramEqualization(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    #taking the V channel of the hsv image
    vlist = hsv[:,:,2]
    
    height,width = vlist.shape
    # height,width, _ = img.shape
        
    hist = createHistogram(vlist)
    
    '''Cumulative Distibution Function
    Number of pixels with intensity less than or equal to "i" intensity
    normalized by N pixels'''
    cdf = list()
    z = 0
    for i in range(len(hist)):
        z = z + (hist[i][1]/(height * width))
        cdf.append(round(z*255))
    
    hnew = np.asarray(cdf)
    
    #Equalize Image based on CDF results
    #----Modifiy Images of video----
    hsv[:,:,2] = hnew[hsv[:,:,2]] 
    
    color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('HSV Equalized',color)
    
    return hist, color


'''Problem 2 Functions'''


'''Problem 3 Functions'''