import cv2
import numpy as np
import os
import time

cam1 = cv2.VideoCapture(0,cv2.CAP_V4L2)
cam2 = cv2.VideoCapture(1,cv2.CAP_V4L2)

fpsReport = 0
startTime = time.time()
while True:
    ret1,frame1 = cam1.read()
    ret2,frame2 = cam2.read()
    # print(frame1.shape,frame2.shape)
        
    if ret1 & ret2 == False:
        print('Fail to open camera')
        
    combCam = np.hstack((frame1,frame2))
    dt = time.time()-startTime
    startTime = time.time()
    fpsReport = .9*fpsReport + .1*dt
    fps = round(1/fpsReport,1)
    cv2.rectangle(combCam,(0,0),(100,50),(0,255,0),2)
    cv2.putText(combCam,'fps:'+str(fps),(0,25),cv2.FONT_HERSHEY_SIMPLEX,.75,(0,255,0),2)
    cv2.imshow('vidoe',combCam)
        
    if cv2.waitKey(1) == ord('q'):
        cam1.release()
        cam2.release()
        cv2.destroyAllWindows()
    
