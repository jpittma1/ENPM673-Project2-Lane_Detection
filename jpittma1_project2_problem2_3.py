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
print("Commencing Problem 2: Straight Lane Detection...")
while (vid.isOpened()):
    count+=1
    success, image = vid.read()
    
    if success:
        '''Straight Lane Detection'''
        # print("Commencing Problem 2: Straight Lane Detection...")

        '''Convert to grayscale'''
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur= cv2.GaussianBlur(grey, (5,5), 0)
        
        '''Reduce Noise'''
        
        '''Detect edges using canny'''
        edges=cv2.Canny(blur, 100,200)
        
        if count==2:
            plt.imshow(edges)
            plt.savefig("edges.png")
        
        img_plus_edges=image.copy()
        
        if problem_2 == True:
            out2.write(img_plus_edges)
        
        '''Apply canny mask to image'''
        
        '''Find coordinates of lane edges'''
        
        '''Fit coordinates into canny image and color'''
        #Green for Solid, Red for Dashed
        
        print("count is ", count)
        # count+=1
        
        # if cv2.waitKey(1) & 0xFF == ord('q'):             
        #     break
    else:
        break


print("Completed Problem 2: Straight Lane Detection!!")
vid.release()
if problem_2 == True:
    out2.release()
cv2.destroyAllWindows()


'''Problem 3: Predict Turns'''
vid=cv2.VideoCapture('challenge.mp4')

vid.set(1,start)
count = start

if problem_3 == True:
    videoname_4=('jpittma1_proj2_problem3')
    out3 = cv2.VideoWriter(str(videoname_4)+".avi", fourcc, fps_out, (1920, 1080))


count = 0
while (vid.isOpened()):
    count+=1
    success, image = vid.read()
    
    if success:
        '''Predict Turns'''
        print("Commencing Problem 3: Predict Turns...")

        
        print("count is ", count)
        # count+=1
        
        # if cv2.waitKey(1) & 0xFF == ord('q'):             
        #     break
    else:
        break


print("Completed Problem 3: Predict Turns!!")
vid.release()
if problem_3 == True:
    out3.release()
cv2.destroyAllWindows()
plt.close('all')