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
    fps =   3
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
    
    #convert heightxwidth from 370,1224 to 480x640so divisible by 8
    cap=cv2.VideoCapture(str(videoname)+".avi")
    size = (640, 480)
    # size = (1280, 720)
    video_new = cv2.VideoWriter(str(videoname)+".avi",fourcc, fps, size)
    
    while True:
        ret,frame_new=cap.read()
        if ret==True:
            b=cv2.resize(frame_new,size,fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
            video_new.write(b)
        else:
            break
    video_new.release()
    video.release()
    cap.release()
    cv2.destroyAllWindows()
    
    return video_new

def createHistogramFigure(img, name):
    
    plt.hist(img.ravel(),256,[0,256])
    # plt.show()
    plt.savefig(name)
    print("Histogram figure saved as: ", name)
    
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

    # cv2.imshow('HSV Equalized',color)
    
    return hist, color


def conductAdaptiveHistogramEqualization(image):
    # img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # print("img.shape: ",img.shape)
    # img=image.copy
    
    '''Define Clip value and search window'''
    clipLimit=0.001 
    nrBins=256
    
    kernel_size = tuple([max(s // 8, 1) for s in image.shape])
    # kernel_size = tuple([max(s // 8, 1) for s in img.shape])
    # print("kernel size: ", kernel_size) #60,80, 1
    
    kernel_size = [int(k) for k in kernel_size]
    # print("kernel size: ", kernel_size) #60,80, 1

    # grayscaleLevels=7800   #as many grayscale levels as possible; this gave me best results
    grayscaleLevels=2**14   #as many grayscale levels as possible; this gave me best results
    # grayscaleLevels=height*width
    ndim = image.ndim
    dtype = image.dtype

    # pad the image such that the shape in each dimension
    # - is a multiple of the kernel_size and
    # - is preceded by half a kernel size
    pad_start_per_dim = [k // 2 for k in kernel_size]

    pad_end_per_dim = [(k - s % k) % k + int(np.ceil(k / 2.))
                       for k, s in zip(kernel_size, image.shape)]

    image = np.pad(image, [[p_i, p_f] for p_i, p_f in
                           zip(pad_start_per_dim, pad_end_per_dim)],
                   mode='reflect')

    # determine gray value bins
    bin_size = 1 + grayscaleLevels // nrBins
    lut = np.arange(grayscaleLevels, dtype=np.min_scalar_type(grayscaleLevels))
    lut //= bin_size

    image = lut[image]

    # calculate graylevel mappings for each contextual region
    # rearrange image into flattened contextual regions
    ns_hist = [int(s / k) - 1 for s, k in zip(image.shape, kernel_size)]
    hist_blocks_shape = np.array([ns_hist, kernel_size]).T.flatten()
    hist_blocks_axis_order = np.array([np.arange(0, ndim * 2, 2),
                                       np.arange(1, ndim * 2, 2)]).flatten()
    
    hist_slices = [slice(k // 2, k // 2 + n * k)
                   for k, n in zip(kernel_size, ns_hist)]
    
    hist_blocks = image[tuple(hist_slices)].reshape(hist_blocks_shape)
    hist_blocks = np.transpose(hist_blocks, axes=hist_blocks_axis_order)
    hist_block_assembled_shape = hist_blocks.shape
    hist_blocks = hist_blocks.reshape((np.product(ns_hist), -1))

    # Calculate actual clip limit
    if clipLimit > 0.0:
        clim = int(np.clip(clipLimit * np.product(kernel_size), 1, None))
    else:
        # largest possible value, i.e., do not clip (AHE)
        clim = np.product(kernel_size)

    # smallestClip=np.product(kernel_size)
    # print("smallest clip, ", smallestClip)
    # print("current clip is ", int(np.clip(clipLimit * np.product(kernel_size), 1, None)))
    # clim=40 #openCV is 40 by default, supposedly
    
    hist = np.apply_along_axis(np.bincount, -1, hist_blocks, minlength=nrBins)
    
    #Clip_histogram
    clipped_hist = np.apply_along_axis(clipHistogram, -1, hist, clip_limit=clim)
    
    '''Create look-up table (LUT) is used to convert the dynamic range
    of the input image into the desired output dynamic range.'''
    #Map Histogram
    hist_map = createLUT(clipped_hist, 0, grayscaleLevels - 1, np.product(kernel_size))
    hist_map = hist_map.reshape(hist_block_assembled_shape[:ndim] + (-1,))

    # duplicate leading mappings in each dimension
    map_array = np.pad(hist_map,
                       [[1, 1] for _ in range(ndim)] + [[0, 0]],
                       mode='edge')

    '''Perform multilinear interpolation of graylevel mappings'''
    '''To remove artifacts after Adaptive Histogram Equalization'''
    # rearrange image into blocks for vectorized processing
    ns_proc = [int(s / k) for s, k in zip(image.shape, kernel_size)]
    blocks_shape = np.array([ns_proc, kernel_size]).T.flatten()
    blocks_axis_order = np.array([np.arange(0, ndim * 2, 2),
                                  np.arange(1, ndim * 2, 2)]).flatten()
    blocks = image.reshape(blocks_shape)
    blocks = np.transpose(blocks, axes=blocks_axis_order)
    blocks_flattened_shape = blocks.shape
    blocks = np.reshape(blocks, (np.product(ns_proc),
                                 np.product(blocks.shape[ndim:])))

    # calculate interpolation coefficients
    coeffs = np.meshgrid(*tuple([np.arange(k) / k
                                 for k in kernel_size[::-1]]), indexing='ij')
    coeffs = [np.transpose(c).flatten() for c in coeffs]
    inv_coeffs = [1 - c for dim, c in enumerate(coeffs)]

    # sum over contributions of neighboring contextual
    # regions in each direction
    result = np.zeros(blocks.shape, dtype=np.float32)
    for iedge, edge in enumerate(np.ndindex(*([2] * ndim))):

        edge_maps = map_array[tuple([slice(e, e + n)
                                     for e, n in zip(edge, ns_proc)])]
        edge_maps = edge_maps.reshape((np.product(ns_proc), -1))

        # apply map
        edge_mapped = np.take_along_axis(edge_maps, blocks, axis=-1)

        # interpolate
        edge_coeffs = np.product([[inv_coeffs, coeffs][e][d]
                                  for d, e in enumerate(edge[::-1])], 0)

        result += (edge_mapped * edge_coeffs).astype(result.dtype)

    result = result.astype(dtype)

    # rebuild result image from blocks
    result = result.reshape(blocks_flattened_shape)
    blocks_axis_rebuild_order =\
        np.array([np.arange(0, ndim),
                  np.arange(ndim, ndim * 2)]).T.flatten()
    result = np.transpose(result, axes=blocks_axis_rebuild_order)
    result = result.reshape(image.shape)

    # undo padding
    unpad_slices = tuple([slice(p_i, s - p_f) for p_i, p_f, s in
                          zip(pad_start_per_dim, pad_end_per_dim,
                              image.shape)])
    result = result[unpad_slices]
    
    # result=(img_CLAHE).astype(np.uint8)
    # color = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    
    return hist, clipped_hist, result
    # return hist, clipped_hist, color
    # return result

def clipHistogram(hist, clip_limit):
    """Perform clipping of the histogram and redistribution of bins.
    The histogram is clipped and the number of excess pixels is counted.
    Afterwards the excess pixels are equally redistributed across the
    whole histogram
    """
    # calculate total number of excess pixels
    excess_mask = hist > clip_limit
    excess = hist[excess_mask]
    n_excess = excess.sum() - excess.size * clip_limit
    hist[excess_mask] = clip_limit

    # Second part: clip histogram and redistribute excess pixels in each bin
    bin_incr = n_excess // hist.size  # average binincrement
    upper = clip_limit - bin_incr  # Bins larger than upper set to cliplimit

    low_mask = hist < upper
    n_excess -= hist[low_mask].size * bin_incr
    hist[low_mask] += bin_incr

    mid_mask = np.logical_and(hist >= upper, hist < clip_limit)
    mid = hist[mid_mask]
    n_excess += mid.sum() - mid.size * clip_limit
    hist[mid_mask] = clip_limit

    while n_excess > 0:  # Redistribute remaining excess
        prev_n_excess = n_excess
        for index in range(hist.size):
            under_mask = hist < clip_limit
            step_size = max(1, np.count_nonzero(under_mask) // n_excess)
            under_mask = under_mask[index::step_size]
            hist[index::step_size][under_mask] += 1
            n_excess -= np.count_nonzero(under_mask)
            if n_excess <= 0:
                break
        if prev_n_excess == n_excess:
            break

    return hist

def createLUT(histogram, min_value, max_value, pixels):
    '''Cumulutative sum of histogram bins'''
    out = np.cumsum(histogram, axis=-1).astype(float)
    out *= (max_value - min_value) / pixels
    out += min_value
    np.clip(out, a_min=None, a_max=max_value, out=out)

    return out.astype(int)


'''Problem 3 Functions'''
#  def solveHomographyAndWarp(img):
     
#      return new_img
 
###--Solves for histogram and max of each column/lane---###
def histogram(img):
    
    hist_vals = list()
    index = list()

    for i in range(img.shape[1]):
        z = np.where(img[:,i] > 0)
        hist_vals.append(len(z[0]))
        index.append(i)
        
    tmp_1 = hist_vals[:120]
    tmp_2 = hist_vals[120:]
    max1 = max(tmp_1)
    max2 = max(tmp_2)
    col1 = hist_vals.index(max1)
    col2 = hist_vals.index(max2)

    bins=200
    plt.plot(index,hist_vals, bins)
    plt.show()
    return col1, col2