import cv2
import numpy as np 
from threading import Thread
import time

class vStream():
    def __init__(self,src,width,height):
        self.capture = cv2.VideoCapture(src,cv2.CAP_V4L2) # Very important,add cv2.CAP_V4L2
        self.width = width
        self.height = height
        self.thread = Thread(target=self.update,args=())
        self.thread.daemon = True
        self.thread.start()
        
    def update(self):
        while True:
            ret,self.frame = self.capture.read()
            if ret == False:
                print('No frame')
            else:
                self.frame = cv2.resize(self.frame,(self.width,self.height))
        
    def getFrame(self):
        return self.frame
    
cam0 = vStream(0,640,480)
cam1 = vStream(1,640,480)
dtav = 0
timeStamp = time.time()
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    try:
        print('0')
        myFrame0 = cam0.getFrame()
        myFrame1 = cam1.getFrame()
        combinFrame = np.hstack((myFrame0,myFrame1))
        dt = time.time()-timeStamp
        timeStamp = time.time()
        dtav = .9*dtav + .1*dt
        fps = 1/dtav
        cv2.rectangle(combinFrame,(0,0),(120,30),(0,0,255),2)
        cv2.putText(combinFrame,'fps:'+str(round(fps,1)),(0,25),font,.75,(0,0,255),2)
        cv2.imshow('frame',combinFrame)
        
    except:
        print('frame not available')

    if cv2.waitKey(1) == ord('q'):
        cam0.capture.release()
        cam1.capture.release()
        cv2.destroyAllWindows()
        exit(1)
        break