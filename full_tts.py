import cv2 
import numpy as np
import argparse
import cv2
import numpy as np
import time
# import tesserocr as tr
from PIL import Image
import os
import scipy.misc as smp
import json
#from pytesser import *
import pprint
import imutils
import pytesseract
# from gtts import gTTS
from espeak import espeak

       

while(1):

        cam = cv2.VideoCapture(1)
        ret, image = cam.read()

        if ret:
            cv2.imshow('SnapshotTest',image)
            cv2.waitKey(0)
            cv2.destroyWindow('SnapshotTest')
            #cv2.imwrite('3.jpg',image)
        cam.release()
        #image = cv2.imread('3.jpg', -1)
        image = cv2.resize(image, (800,600), interpolation = cv2.INTER_AREA)     

        # cv2.imshow('shadows_o', image)
        # cv2.waitKey(0)

        rgb_planes = cv2.split(image)

        result_planes = []
        result_norm_planes = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = np.zeros((diff_img.shape[0],diff_img.shape[1], 1), dtype=np.uint8)

            cv2.normalize(diff_img, norm_img , 0, 255,cv2.NORM_MINMAX,dtype=cv2.CV_8UC1)
            result_planes.append(diff_img)
            result_norm_planes.append(norm_img)


        result = cv2.merge(result_planes)


        result_norm = cv2.merge(result_norm_planes)
        # result = 255 - result
        # cv2.imshow('shadows_out.png', result)
        # cv2.waitKey(0)
        # cv2.imshow('shadows', result_norm)
        # cv2.waitKey(0)
        gray = cv2.cvtColor(result_norm , cv2.COLOR_BGR2GRAY)
        # cv2.imshow('sd',gray)
        # cv2.waitKey(0)



        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5))
        # gray = clahe.apply(gray)

        # cv2.imshow("cla",gray)
        # cv2.waitKey(0)


        alpha = 2.0 # Simple contrast control//float(1-3)
        beta = -80  # Simple brightness control.//int(0-100)


        # smallest = np.amin(gray)
        # biggest = np.amax(gray)
        # print(smallest)
        # print(biggest)
        # input_range = biggest - smallest
        # wanted_output= 255;
        # alpha = 255 / input_range


        # beta = -smallest* alpha
        crude = image.copy()
        # gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        # cv2.imshow("abeta",gray)

        # cv2.waitKey(0)

        processed_image = np.int32(gray)
        #print(processed_image)
        processed_image = processed_image*0.02

        out = np.exp(processed_image)
        out = out - 1

        cv2.normalize(out, out, 0, 255, cv2.NORM_MINMAX)
        # cv2.imshow("norm",out)
        # cv2.waitKey(0)

        cv2.convertScaleAbs(out,out)
        thresh = 255 - out
        thresh = np.uint8(thresh)
        # cv2.imshow("expo",thresh)
        # cv2.waitKey(0)

        ret,thresh = cv2.threshold(thresh,200,255,cv2.THRESH_BINARY)
        #cv2.imwrite("result.jpg",out)
        # cv2.imshow("out",thresh)
        # cv2.waitKey(0)


        mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)

        #dilation
        # kernel = np.ones((1,1), np.uint8)
        ##########
        ######### boxes
        # kernel = np.ones((10,15), np.uint8) 88
        kernel = np.ones((7,7), np.uint8)
        img_dilation = cv2.dilate(thresh, kernel, iterations=1)#thresh if thresholding with black on white
        # cv2.imshow('dilated', img_dilation)
        # cv2.waitKey(0)

        vis = mask.copy()

        #find contours
        __,ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
         
        #sort contours
        sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])


        angle = 0
        k=0
        for i, ctr in enumerate(sorted_ctrs):
            # x, y, w, h = cv2.boundingRect(ctr)
            k = k+1
            rect = cv2.minAreaRect(ctr)
            # box = cv2.boxPoints(rect)
            # box = np.int0(box)
            # cv2.drawContours(gray,[box],0,(0,0,0),2)
            (w,h) = rect[1]
            (x,y) = rect[0]
            if (w < 300 and h < 300) and (w>20 and h>20):

              ang = rect[2]
              if ang < -45:
                ang = -(90 + ang)
              angle += ang
              
            

        angle = 2*angle/k
        # print(".................")
        # print(angle)
        # cv2.imshow("cont",gray)
        # cv2.waitKey(0)
        #angle = -8.7
        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(gray, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        image = cv2.warpAffine(image, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        crude = cv2.warpAffine(crude, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # draw the correction angle on the image so we can validate it
        #cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
         # (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        gray = rotated.copy()
        newim = image.copy() 
        newgray = gray.copy()
        # cv2.imshow("deskewed",gray)
        # cv2.waitKey(0)



        ######################################################3skew
        processed_image = np.int32(gray)
        #print(processed_image)
        processed_image = processed_image*0.02

        out = np.exp(processed_image)
        out = out - 1

        cv2.normalize(out, out, 0, 255, cv2.NORM_MINMAX)
        # cv2.imshow("norm",out)
        # cv2.waitKey(0)

        cv2.convertScaleAbs(out,out)
        thresh = 255 - out
        thresh = np.uint8(thresh)
        # cv2.imshow("expo",thresh)
        # cv2.waitKey(0)

        ret,thresh = cv2.threshold(thresh,200,255,cv2.THRESH_BINARY)
        #cv2.imwrite("result.jpg",out)
        # cv2.imshow("threshagain",thresh)
        # cv2.waitKey(0)


        mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)

        #dilation
        # kernel = np.ones((1,1), np.uint8)
        ##########
        ######### boxes
        # kernel = np.ones((10,15), np.uint8) 88
        kernel = np.ones((7,7), np.uint8)
        img_dilation = cv2.dilate(thresh, kernel, iterations=1)#thresh if thresholding with black on white
        # cv2.imshow('dilated', img_dilation)
        # cv2.waitKey(0)

        vis = mask.copy()

        #find contours
        __,ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
         
        #sort contours
        sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])


        for i, ctr in enumerate(sorted_ctrs):
            # Get bounding box
            x, y, w, h = cv2.boundingRect(ctr)
         
            # Getting ROI
            # roi = image[y:y+h, x:x+w]
         
            # # show ROI
            # #cv2.imshow('segment no:'+str(i),roi)
            if (w < 300 and h < 150) and (w>20 and h>20):

              cv2.rectangle(gray,(x,y),( x + w, y + h ),(0,0,0),2)
              # #cv2.waitKey(0)
              cv2.rectangle(mask, (x,y), (x+w,y+h), color=(255,255,255))

         
            # if w > 15 and h > 15:
            #       cv2.imwrite('C:\\Users\\Link\\Desktop\\output\\{}.png'.format(i), roi)

         
         
        # cv2.imshow('draw',gray)
        # cv2.waitKey(0)

        # cv2.imshow('mask',mask)
        # cv2.waitKey(0)

        __,contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        # out = cv2.drawContours(vis, contours , -1, (255,255,255), 1)
        # cv2.imshow('contnew',out)
        # cv2.waitKey()



        ####################################$##################################################
        #drawing the finger tip
        # new =cv2.imread('3.jpg')
        # new = cv2.resize(new, (800,600), interpolation = cv2.INTER_AREA)        
        # cv2.imshow('new',new)
        # cv2.waitKey()

        hsv= cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # cv2.imshow('hsv',hsv)
        # cv2.waitKey()

        lower = np.array([0, 48, 80], dtype = "uint8")
        upper = np.array([20, 255, 255], dtype = "uint8")
        tip = cv2.inRange(hsv, lower, upper)

        # cv2.imshow('contnew',tip)
        # cv2.waitKey()

        cnts = cv2.findContours(tip, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        c = max(cnts, key=cv2.contourArea)
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        # print extTop
        w,h=extTop

        cv2.line(vis,(w,0),(w,h),(255,0,0),5)     ###########drawing line
        # cv2.imshow('wind', vis)
        # cv2.waitKey(0) 

        #accessing the contour through which the line passes
        ##################################################
        i=0 #no of contours through which line passes
        # print "for loop"
        cont=[]
        j=0 #total no of contours
        k=0 #x and y coordinates
        for cn in contours:
          #print cn
          j=j+1
          for p in cn:
             # print "second loop"
             for k in p:
              #print "second loop"
              x=k[0]
              y=k[1]
              if x==w :
                cont.append(cn)
                # print j
                # print x
                # print y
                # # cv2.drawContours(plane, contours, j-1, (255,255,255), 3)
                # cv2.imshow('cont',plane)
                # cv2.waitKey(0)
                i=i+1 #number of contours through which the line passes
                k=1
                break
              else:#to avoid detecting the countour twice
                k=0
             if k==1:
              break

          if k==1:
            break    
        ############################
        #finding the contour st minimum distance
        l=0
        # print i
        # print j
        minm = 0
        for cnt in cont:
          l=l+1
          M = cv2.moments(cnt)
          if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/(0.5+M['m00']))
            dist = abs(h-cy)
            if minm==0:
              minm = dist
            
            if dist<=minm:
              minm = dist
              i = l 
              c = cnt

####################333second largest############################
        # for cnt in cont:
        #   l=l+1
        #   M = cv2.moments(cnt)
        #   if M['m00'] != 0:
        #     cx = int(M['m10']/M['m00'])
        #     cy = int(M['m01']/(0.5+M['m00']))
        #     dist = abs(h-cy)
        #     if minm==0:
        #       minm = dist
            
        #     if dist<=minm:
        #       minm = dist
        #       i = l 
        #       c = cnt

####################333second largest############################


        # img = cv2.imread('3.jpg')
        # img = cv2.resize(img, (800,600), interpolation = cv2.INTER_AREA)        
        cv2.drawContours(image, cont, i-1, (255,0,0), 3)
        # cv2.imshow('cont',image)
        # cv2.waitKey(0)
        # # print cnt

        # print "111111111111111111111111111111111111111111111"
        # print angle
        ###################################3 


        #########crop
        x,y,w,h = cv2.boundingRect(c)
        # img = cv2.imread('3.jpg')
        # img = cv2.resize(img, (800,600), interpolation = cv2.INTER_AREA)        
        new_img=newgray[y:y+h,x:x+w]#####new_img=thresh[y:y+h,x:x+w]
        # cv2.imwrite('crop.jpg',new_img)
        # cv2.imshow('np',new_img)
        # cv2.waitKey(0)

        
        if new_img is None:
            espeak.synth("No word detected")

          # print("No word detected")
            contiue
      
        # ##################################################
        #paste
        # roi=img[y:y+h,x:x+w]
        # roi.shape
        # # img2=img.copy()

        # pic[300:(300+roi.shape[0]), 400:(400+roi.shape[1])]=roi

        mask = np.ones((image.shape[0], image.shape[1], 1), dtype=np.uint8) 

        # cv2.imwrite('black.jpg',mask)

        # try:
        #         #Relative Path
        #         #Image on which we want to paste
        #         img = Image.open("black.jpg") 
                 
        #         #Relative Path
        #         #Image which we want to paste
        #         img2 = Image.open("crop.jpg") 
        #         img.paste(img2, (50, 50))
                 
        #         #Saved in the same relative location
        #         img.save("piccc.jpg")

        # # new_img=img[y:y+h,x:x+w]
        # # img.paste(new, (50, 50))


        # imagg= cv2.imread('piccc.jpg')

        # cv2.imshow('np',imagg)
        # cv2.waitKey(0) 
        #####################################
        #ROI
        mask = cv2.bitwise_not(mask)


        mask[300:(300+new_img.shape[0]), 400:(400+new_img.shape[1])]=new_img
        # cv2.imshow('np',mask)
        # cv2.waitKey(0)

        # gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        ret,thresh = cv2.threshold(mask,130,255,0)
        cv2.imshow('np',thresh)
        cv2.waitKey(0)



        text = pytesseract.image_to_string(thresh)
        print(text)
        cv2.destroyAllWindows()
        file = open("testfile.txt", "w") 
        file.write(text) 
        file.close() 

        file= open("testfile.txt","r")
        # tts = gTTS(text=file.read(), lang='en')
        # tts.save("good.mp3")

        # 
        # 
        espeak.synth(file.read())
        time.sleep(2)
