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
from numpy import histogram_bin_edges, linalg as LA
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

def createHistogramFigure(val, name):
    
    plt.hist(val,bins=255)
    # plt.show()
    plt.savefig(name)

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

'''To remove artifacts after Adaptive Histogram Equalization'''
def bilinearInterpolation(subBin,LU,RU,LB,RB,subX,subY):
    tmp_img = np.zeros(subBin.shape)
    num = subX*subY
    for i in range(subX):
        inverseI = subX-i
        for j in range(subY):
            inverseJ = subY-j
            val = subBin[i,j].astype(int)
            tmp_img[i,j] = np.floor((inverseI*(inverseJ*LU[val] + j*RU[val])+ i*(inverseJ*LB[val] + j*RB[val]))/float(num))
    return tmp_img

def conductAdaptiveHistogramEqualization(img):
    '''Define Clip value and search window'''
    clipLimit=50 #openCV is 40 by default, supposedly
    nrBins=256
    height,width, _ = img.shape
    
    
    #Search Window size (openCV default is 8x8, supposedly)
    x_size=8
    y_size=8
    windowPixels=x_size*y_size
    
    x_regions=np.ceil(height/x_size).astype(int)    #47
    y_regions=np.ceil(width/y_size).astype(int)     #153
    # print("x_regions: ", x_regions, ", y_regions: ", y_regions)
    
    img_CLAHE=np.zeros(img[:,:,0].shape)
    
    '''Create look-up table (LUT) is used to convert the dynamic range
    of the input image into the desired output dynamic range.'''
  
    binSize = np.floor(256/float(nrBins))
    # print("binSize is: ", binSize)  #1.0
    
    LUT = np.floor((np.arange(0,256))/float(binSize))
    # print("LUT is:  ", LUT)
    
    tmp = LUT[img]
    # print("temp bins from LUT shape is: ", tmp.shape)   #370,1224,3
    
    '''Make Histogram for each window'''
    hist=np.zeros((x_regions, y_regions, nrBins))
    # print("Hist shape is: ", hist.shape)    #47,153,256
    
    for i in range(x_regions):  #0->47
        for j in range(y_regions):   #0->153
            tmp_bin = tmp[i*x_size:(i+1)*x_size, j*y_size: (j+1)*y_size].astype(int)
            # print("shape of tmp_bin is: ", tmp_bin.shape)   #8, 8, 3
            for k in range(x_size):  #0->8
                for l in range(y_size): #0->8
                    hist[ i,j, tmp_bin[k,l] ] += 1  #ERROR!!!!
                    # print("Current histogram is: ", hist[ i,j, tmp_bin[k,l] ])
                    #TODO: index 2 is out of bounds for axis 0 with size 2
    
    print("Adaptive Histogram, pre-clipped is: ", hist)
    
    '''Clip Histogram'''
    hist_clipped=hist
    
    if hist_clipped==hist:
        print("copy successful!!")
    
    for i in range(x_regions):
        for j in range(y_regions):
            total_excess=0
            
            for nr in range(nrBins):
                excess=hist_clipped[i,j,nr]-clipLimit
                if excess>0:
                    total_excess += excess
            
            '''Distribute clipped pixels uniformly to bins'''
            binIncrease = total_excess/nrBins
            new_top = clipLimit - binIncrease
            
            for nr in range(nrBins):
                if hist_clipped[i,j,nr]>clipLimit:
                    hist_clipped[i,j,nr]=clipLimit
                else:
                    if hist_clipped[i,j,nr]>new_top:
                        total_excess += new_top - hist_clipped[i,j,nr]
                        hist_clipped[i,j,nr] = clipLimit
                    else: 
                        total_excess -= binIncrease
                        hist_clipped[i,j,nr] += binIncrease
            
            if total_excess > 0:
                stepSize = max(1,np.floor(1+total_excess/nrBins))
                for nr in range(nrBins):
                    total_excess -= stepSize
                    hist_clipped[i,j,nr] += stepSize
                    if total_excess < 1:
                        break
            
    '''Create map from Histogram for interpolation'''
    map_ = np.zeros((x_regions,y_regions,nrBins))
    #print(map_.shape)
    scale = 255/float(windowPixels)
    for i in range(x_regions):
        for j in range(y_regions):
            sum_ = 0
            for nr in range(nrBins):
                sum_ += hist[i,j,nr]
                map_[i,j,nr] = np.floor(min(sum_ * scale,255))
    
    '''Bilinear interpolation
    xU=upper X, yL=left y, yR=right Y, xB=bottom X
    Moves through the window of pixels updating clahe_image as it goes'''
    xI=0    #interpolation X
    for i in range(x_regions+1):
        if i==0:
            subX = int(x_size/2)
            xU = 0
            xB = 0
        elif i==x_regions:
            subX = int(x_size/2)
            xU = x_regions-1
            xB = x_regions-1
        else:
            subX = x_size
            xU = i-1
            xB = i
    
        yI = 0 #interpolation Y
        for j in range(y_regions+1):
            if j==0:
                subY = int(y_size/2)
                yL = 0
                yR = 0
            elif j==y_regions:
                subY = int(y_size/2)
                yL = y_regions-1
                yR = y_regions-1
            else:
                subY = y_size
                yL = j-1
                yR = j
            UL = map_[xU,yL,:]
            UR = map_[xU,yR,:]
            BL = map_[xB,yL,:]
            BR = map_[xB,yR,:]
        
            print("subX is ", subX)
            print("subY is ", subY)
            
            claheBins=tmp[xI:xI+subX, yI:yI+subY]
        
            interpolate_image=bilinearInterpolation(claheBins,UL,UR,BL,BR,subX,subY)
            
            img_CLAHE[xI:xI+subX, yI:yI+subY] = interpolate_image
            
            yI += subY
        xI += subX
    
    #TODO: Gamma adjust??
    #setting the gamma value, increased values may cause noise
    # gamma = 1.4
    # def adjust_gamma(image, gamma=1.0):
    #     invGamma = 1.0 / gamma
    # table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)])
    # return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))
    # cl1= adjust_gamma(cl1, gamma=gamma)
    # #adding the last V layer back to the HSV image
    # img2hsv[:,:,2] = cl1
    
    # improved_image = cv2.cvtColor(img2hsv, cv2.COLOR_HSV2BGR)
    
    return hist, hist_clipped, img_CLAHE


'''Problem 2 Functions'''


'''Problem 3 Functions'''