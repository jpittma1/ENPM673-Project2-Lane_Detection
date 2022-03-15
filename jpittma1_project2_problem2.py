#!/usr/bin/env python3

#ENPM673 Spring 2022
#Section 0101
#Jerry Pittman, Jr. UID: 117707120
#jpittma1@umd.edu
#Project #2

#********************************************
#Requires the following in same folder to run:
# 1) "functions.py"
# 2) "whiteline.mp4"
# 3) "challenge.mp4"
#********************************************
from functions import *

##----to toggle making Videos----##
problem_2 = False
problem_3 = False
#####################


###---Values for making videos----##
if problem_2 == True or problem_3 == True:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps_out = 3 #Ensure matches night_drive_video!!
    print("Making a video...this will take some time...")


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
    
count = 0
while (vid.isOpened()):
    count+=1
    success, image = vid.read()
    
    if success:
        '''Histogram Equalization'''
        

        
        print("count is ", count)
        # count+=1
        
        # if cv2.waitKey(1) & 0xFF == ord('q'):             
        #     break
    else:
        break

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