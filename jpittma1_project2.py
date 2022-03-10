#!/usr/bin/env python3

#ENPM673 Spring 2022
#Section 0101
#Jerry Pittman, Jr. UID: 117707120
#jpittma1@umd.edu
#Project #2

#********************************************
#Requires the following in same folder to run:
# 1) "functions.py"
# 2) folder "adaptive_hist_data" with 25 images
# 3) "whiteline.mp4"
# 3) "challenge.mp4"
#********************************************
from functions import *

'''Problem 1: Histogram Equalization'''
##----to toggle making Videos----##
problem_1 = False
problem_2 = False
problem_3 = False
#####################


#--Read images
# testudo=cv2.imread('testudo.png')
# cv2.imshow('image', testudo)

#---Values for making videos---
if problem_1 == True or problem_2 == True or problem_3 == True:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps_out = 29
    print("Making a video...this will take some time...")

if problem_1 == True:
    videoname=('jpittma1_proj2_problem1')
    out1 = cv2.VideoWriter(str(videoname)+".avi", fourcc, fps_out, (1920, 1080))


                



if problem_1 == True:
    out1.release()


'''Problem 2: Straight Lane Detection'''
#---Read the video, save a frame
thresHold=180
start=1 #start video on frame 1
vid=cv2.VideoCapture('whiteline.mp4')

vid.set(1,start)
count = start

if (vid.isOpened() == False):
    print('Please check the file name again and file location!')


if problem_2 == True:
    videoname1=('jpittma1_proj2_problem2')
    out2 = cv2.VideoWriter(str(videoname1)+".avi", fourcc, fps_out, (1920, 1080))

vid.release()
if problem_2 == True:
    out2.release()

'''Problem 3: Predict Turns'''
vid=cv2.VideoCapture('challenge.mp4')

vid.set(1,start)
count = start

if problem_3 == True:
    videoname2=('jpittma1_proj2_problem3')
    out3 = cv2.VideoWriter(str(videoname2)+".avi", fourcc, fps_out, (1920, 1080))


vid.release()
if problem_3 == True:
    out3.release()
cv2.destroyAllWindows()