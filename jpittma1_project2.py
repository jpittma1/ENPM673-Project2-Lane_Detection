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

'''Problem 1: Histogram Equalization
Goal: Enhance constrast and improve visual appearance of video sequence'''

##----to toggle making Videos----##
problem_1 = False
problem_2 = False
problem_3 = False
#####################

#---Read images "video sequence"----
path = './adaptive_hist_data/'
print("Converting provided images into a video...")
night_drive_video=convertImagesToMovie(path)
# night_drive_video.release()

thresHold=180
start=1 #start video on frame 1
vid=cv2.VideoCapture('night_drive.avi')

vid.set(1,start)

if (vid.isOpened() == False):
    print('Please check the file name again and file location!')

###---Values for making videos----##
if problem_1 == True or problem_2 == True or problem_3 == True:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps_out = 3
    print("Making a video...this will take some time...")

if problem_1 == True:
    videoname_1=('jpittma1_proj2_problem1_hist')
    out1a = cv2.VideoWriter(str(videoname_1)+".avi", fourcc, fps_out, (1224, 370))

    videoname_2=('jpittma1_proj2_problem1_hist_adapt')
    out1b = cv2.VideoWriter(str(videoname_2)+".avi", fourcc, fps_out, (1224, 370))

#Histogram equalization
count = 0
while (vid.isOpened()):
    count+=1
    success, image = vid.read()
    
    if success:
        '''Histogram Equalization'''
        print("Conducting Histogram Equalization...")

        # bins = np.zeros(256)
        bins = np.zeros((256,),dtype=np.float16)
  
        # image_hist=conductHistogramEqualization(image, bins)
        hist, image_hist=conductHistogramEqualization(image)
        # print("hist is ", hist_vals)
        
        if problem_1 == True:
            out1a.write(image_hist)
            print("Frame ",count, "saved")
        
        if count==2:
            createHistogramFigure(hist, 'histogram_frame2.png')
            
            hist_compare = np.vstack((image, image_hist))
            # cv2.imshow('hist_compare', hist_compare)
            
            cv2.imwrite("nightDrive_HistCompare_frame2.jpg", hist_compare)
            cv2.imwrite("nightDrive_Hist_frame2.jpg" , image_hist)
            print("Histogram Equalized image saved as 'nightDrive_Hist_frame2.jpg'")

        '''Adaptive Histogram Equalization'''
        print("Conducting Adaptive Histogram Equalization...")
        #TODO  Adaptive Histogram equalization
        hist_adapt, hist_adapt_clipped, image_adapt_hist=conductAdaptiveHistogramEqualization(image)
        
        if problem_2 == True:
            out1b.write(image_adapt_hist)
        
        if count==2:
            createHistogramFigure(hist_adapt, 'adaptive_histogram_frame2.png')
            createHistogramFigure(hist_adapt_clipped, 'adaptive_histogram_PostClip_frame2.png') 
            hist_adapt_compare = np.vstack((image, image_hist,image_adapt_hist))
            cv2.imwrite("nightDrive_Hist_adapt_Compare_frame2.jpg", hist_adapt_compare)
            cv2.imwrite("nightDrive_AdaptHist_frame%d.jpg" % count, image_adapt_hist)
            print("Adaptive Histogram Equalized image saved as 'nightDrive_AdaptHist_frame2.jpg'")
        
        print("count is ", count)
        # count+=1
        
        # if cv2.waitKey(1) & 0xFF == ord('q'):             
        #     break
    else:
        break
    

night_drive_video.release()
vid.release()
if problem_1 == True:
    out1a.release()
    out1b.release()


'''Problem 2: Straight Lane Detection'''
#Green for Solid, Red for Dashed
#---Read the video, save a frame
thresHold=180
start=1 #start video on frame 1
vid=cv2.VideoCapture('whiteline.mp4')

vid.set(1,start)
count = start

if (vid.isOpened() == False):
    print('Please check the file name again and file location!')


if problem_2 == True:
    videoname_3=('jpittma1_proj2_problem2')
    out2 = cv2.VideoWriter(str(videoname_3)+".avi", fourcc, fps_out, (1920, 1080))

vid.release()
if problem_2 == True:
    out2.release()

'''Problem 3: Predict Turns'''
vid=cv2.VideoCapture('challenge.mp4')

vid.set(1,start)
count = start

if problem_3 == True:
    videoname_4=('jpittma1_proj2_problem3')
    out3 = cv2.VideoWriter(str(videoname_4)+".avi", fourcc, fps_out, (1920, 1080))


vid.release()
if problem_3 == True:
    out3.release()
cv2.destroyAllWindows()