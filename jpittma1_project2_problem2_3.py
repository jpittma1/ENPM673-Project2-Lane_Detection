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

width  = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))   
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

# print('width, height:', width, height)    #(960, 540)

vid.set(1,start)
count = start

if (vid.isOpened() == False):
    print('Please check the file name again and file location!')


if problem_2 == True:
    videoname_3=('jpittma1_proj2_problem2')
    out2 = cv2.VideoWriter(str(videoname_3)+".avi", fourcc, fps_out, (width, height))
    
count = 0
print("Commencing Problem 2: Straight Lane Detection...")
while (vid.isOpened()):
    count+=1
    success, image = vid.read()

    
    if success:
        '''Straight Lane Detection'''
        # print("Commencing Problem 2: Straight Lane Detection...")

        ###---Pre-process the Video Frames---###
        '''Convert to grayscale'''
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        

        '''Crop Frame to remove non-road sections'''
        '''Get Region of Interest (bottom half of movie)'''
        #NOTE: Cropping makes difficult to reapply lane colors on original image
        h,w, _= image.shape
        ROI_h=int(h/2) #cut upper half of image
        img_crop=grey[ROI_h:h, 1:w]
        
        '''Reduce Noise'''
        blur= cv2.medianBlur(grey, 5)
        # blur= cv2.medianBlur(grey, 5, 0)
        # blur= cv2.medianBlur(img_crop, 5, 0)
        # blur= cv2.GaussianBlur(img_crop, (5,5), 0)
        
        ###---Lane Detection---###
        '''Detect edges using canny'''
        #Green for Solid, Red for Dashed
        edges=cv2.Canny(blur, 100,200)
        
        edges_color=cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        linesP_solid = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, 5, minLineLength = 300, maxLineGap=10)
    
        if linesP_solid is not None:
            for i in range(0, len(linesP_solid)):
                l = linesP_solid[i][0]
                cv2.line(edges_color, (l[0], l[1]), (l[2], l[3]), (0,255,0), 3, cv2.LINE_AA)
                cv2.line(image, (l[0], l[1]), (l[2], l[3]), (0,255,0), 3, cv2.LINE_AA)
        
        
        linesP_dashed = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, minLineLength =5, maxLineGap=None)
        
        #####TODO: Remove solid lines from Dashed Lines--##############
        # linesP_dashed_clip=[]
        # if linesP_dashed is not None:
        #     for j in range (0,len(linesP_dashed)):
        #         l = linesP_dashed[j][0]
        #         if l not in linesP_solid:
        #             linesP_dashed_clip.append(j)
        
        
        # if linesP_dashed_clip is not None:
        #     for i in range(0, len(linesP_dashed_clip)):
        #         l = linesP_dashed_clip[i][0]
        #         cv2.line(edges_color, (l[0], l[1]), (l[2], l[3]), (255,0,0), 2, cv2.LINE_AA)
        
        ######################################
                
        if linesP_dashed is not None:
            for i in range(0, len(linesP_dashed)):
                l = linesP_dashed[i][0]
                cv2.line(edges_color, (l[0], l[1]), (l[2], l[3]), (255,0,0), 3, cv2.LINE_AA)
                cv2.line(image, (l[0], l[1]), (l[2], l[3]), (255,0,0), 3, cv2.LINE_AA)
        
        ####----Make images for Report of pipeline steps----######
        if count==2:
            print("Frame 2, Making images for report...")
            plt.imshow(img_crop)
            plt.savefig("img_crop.png")
            plt.imshow(blur)
            plt.savefig("img_medianBlur.png")
            plt.imshow(edges)
            plt.savefig("edges.png")
            
            plt.imshow(image)
            plt.savefig("img_colored.png")
            
            plt.imshow(edges_color)
            plt.savefig("img_edges_colored.png")
        
        img_plus_edges=image.copy()
        
        if problem_2 == True:
            out2.write(img_plus_edges)
        

        # print("count is ", count)

    else:
        break


print("Completed Problem 2: Straight Lane Detection!!")
vid.release()
if problem_2 == True:
    out2.release()
cv2.destroyAllWindows()
plt.close('all')


'''Problem 3: Predict Turns'''
print("Commencing Problem 3: Predict Turns...")
vid=cv2.VideoCapture('challenge.mp4')

width  = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))   
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

vid.set(1,start)
count = start

if problem_3 == True:
    videoname_4=('jpittma1_proj2_problem3')
    out3 = cv2.VideoWriter(str(videoname_4)+".avi", fourcc, fps_out, (width, height))

count = 0
while (vid.isOpened()):
    count+=1
    success, frame = vid.read()
    
    if success:
        ####----Pre-processing----####
        '''Blur (Remove Noise) Image'''
        blur=cv2.medianBlur(frame, 5)
        
        '''Crop Image'''
        #720 height so split in half
        crop_bottom=blur[360:,:]
        crop_top=blur[:360,:]
        
        # recombine = np.concatenate((crop_top,crop_bottom), axis = 0)
        
        '''Find Homography and Warp'''
        warp=solveHomographyAndWarp(crop_bottom)
        
        '''Convert to HLS format 
        (R, G, and B are converted to the floats and scaled to fit the 0 to 1 range.'''
        img_hls=cv2.cvtColor(warp, cv2.COLOR_BGR2HLS)
        # img_hls=cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)
        # img_hls=cv2.cvtColor(crop_bottom, cv2.COLOR_BGR2HLS)
        
        '''Create masks for yellow and white lane markings'''
        yellowLow = np.array([15, 100, 20])
        yellowHigh = np.array([60, 200, 250])
        whiteLow = np.array([0, 180, 0])
        whiteHigh = np.array([255, 255, 255])
        
        yellowMask = cv2.inRange(img_hls, yellowLow, yellowHigh)
        whiteMask= cv2.inRange(img_hls, whiteLow,whiteHigh)
        
        yellow_hls = cv2.bitwise_and(img_hls, img_hls, mask = yellowMask).astype(np.uint8)
        white_hls = cv2.bitwise_and(img_hls, img_hls, mask=whiteMask).astype(np.uint8)
        
        comb_hls = cv2.bitwise_or(white_hls,yellow_hls)
        
        '''Histogram of Intensities'''
        #Lane_l is left lane values, lane_r is right lane values
        # lane_l,lane_r,ind,vals, b = histogram(comb_hls)
        lane_l,lane_r = histogram(comb_hls)
        
        
        '''Find white pixels in max and neighboring histogram columns'''
        try:
            leftpts = np.where(comb_hls[:,lane_l-10:lane_l+11] > 0)
            rightpts = np.where(comb_hls[:,lane_r-10:lane_r+11] > 0)
            
            #calculating the coefficients of the polynomials of left and right lanes
            coeff_left = np.polyfit(leftpts[0], leftpts[1]+(lane_l-10), 2)
            coeff_right = np.polyfit(rightpts[0], rightpts[1]+(lane_r-10), 2)
        except:
            continue
        
        poly_points, left_pts, right_pts = solvePolygon(coeff_left, coeff_right)
        
        '''Plotting the polygon on image'''
        # cv2.fillPoly(crop_bottom, [poly_points], (50, 205, 50))
        # cv2.fillPoly(crop_bottom, [poly_points], (100, 120, 50))    #green
        cv2.fillPoly(warp, [poly_points], (100, 120, 50))    #green
        
        
        '''Lane Mask'''
        shape0=crop_bottom.shape[0]
        shape1=crop_bottom.shape[1]
        
        img_unwarp=unwarpImage(warp, shape0,shape1)
        
        #thresholding the unwarped grayscale image
        newwarp_gray = cv2.cvtColor(img_unwarp, cv2.COLOR_BGR2GRAY)
        
        _, un_thresh = cv2.threshold(newwarp_gray, 0, 250, cv2.THRESH_BINARY_INV)
        
        #extracting only the road from the unwarped image to use as a mask 
        mask_inv = cv2.bitwise_and(crop_bottom, crop_bottom, mask = un_thresh).astype(np.uint8)
        
        #adding the resultant mask with the unwarped image
        result_half = cv2.add(mask_inv, img_unwarp)
        
        '''Re-stitch upper and lower halfs of images/frames'''
        recombine = np.concatenate((crop_top,result_half), axis = 0)
        
        img_final=recombine.copy()
        
        '''Predict Direction'''
        # curveRadius=solveRadiusOfCurve(coeff_left,coeff_right, left_pts, right_pts)
        curveRadius = ((1+((2*coeff_right[0]*right_pts[50][1])+coeff_right[1])**2)**(3/2))/abs(2*coeff_right[0])
        
        img_center=int(warp.shape[1]/2)
        
        #Add Text to frame indicating curvature/turn
        addCurveDirectionText(img_center, lane_l, lane_r, img_final, curveRadius)
        
        if count==2:
            print("Frame 2, Making images for report...")
           
            plt.imshow(blur)
            plt.savefig("prob3_medianBlur.png")
            
            plt.imshow(yellow_hls)
            plt.savefig("prob3_HSV yellow.png")
            plt.imshow(white_hls)
            plt.savefig("prob3_HSV white.png")
            plt.imshow(comb_hls)
            plt.savefig("prob3_HSV combined.png")
            
            plt.imshow(warp)
            plt.savefig("prob3_warped.png")
            
            plt.imshow(img_unwarp)
            plt.savefig("prob3_unwarped.png")
            

            plt.imshow(recombine)
            plt.savefig("prob3_polyLines.png")
            
            plt.imshow(img_final)
            plt.savefig("prob3_final_result.png")
            
            # plt.imshow(edges)
            # plt.savefig("edges.png")
            # plt.imshow(img_crop)
            # plt.savefig("img_crop.png")
            # plt.imshow(image)
            # plt.savefig("img_colored.png")
            
            # plt.imshow(edges_color)
            # plt.savefig("img_edges_colored.png")
        
        # print("count is ", count)
    
    
        if problem_3 == True:
                out3.write(img_final)
                # out3.write(frame)

    else:
        break


print("Completed Problem 3: Predict Turns!!")
vid.release()
if problem_3 == True:
    out3.release()
cv2.destroyAllWindows()
plt.close('all')