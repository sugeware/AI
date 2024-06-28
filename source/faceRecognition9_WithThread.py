import cv2
import numpy as np 
from threading import Thread
import time
import face_recognition
import pickle

class vStream():
    def __init__(self,src,width,height):
        self.capture = cv2.VideoCapture(src,cv2.CAP_V4L2) # Very important,add cv2.CAP_V4L2
        # self.capture = cv2.VideoCapture(src) 
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
    
# cam_url = 'rtsp://admin:Su619865277@192.168.110.166/ch1-s1/tcp'   
# cam0 = vStream(cam_url,640,480)
cam0 = vStream(0,640,480)
cam1 = vStream(1,640,480)

dtav = 0
timeStamp = time.time()
font = cv2.FONT_HERSHEY_SIMPLEX
scaleFactor = .3
Names = []
Encodings = []

with open('/home/jetson/Desktop/AI/models/train2.pkl','rb') as f:
    Names = pickle.load(f)
    Encodings = pickle.load(f)
    
# while True:
#     frame1 = cam0.getFrame()
#     cv2.imshow('frame',frame1)
    
while True:
    try:
        print('0')
        myFrame0 = cam0.getFrame()
        # combinFrame = cam0.getFrame()
        myFrame1 = cam1.getFrame()
        combinFrame = np.hstack((myFrame0,myFrame1))
        smallFrame = cv2.resize(combinFrame,(0,0),fx = scaleFactor,fy = scaleFactor)
        frameRGB = cv2.cvtColor(smallFrame,cv2.COLOR_BGR2RGB)
        facePositions = face_recognition.face_locations(frameRGB,model='cnn')
        allEncodings = face_recognition.face_encodings(frameRGB,facePositions)
        for (top,right,bottom,left),face_encoding in zip(facePositions,allEncodings):
            name = 'UnknownPerson'
            matchs = face_recognition.compare_faces(Encodings,face_encoding)
            if True in matchs:
                first_matchs_index = matchs.index(True)
                name = Names[first_matchs_index]
            top = int(top/scaleFactor)
            right = int(right/scaleFactor)
            bottom = int(bottom/scaleFactor)
            left = int(left/scaleFactor)
            cv2.rectangle(combinFrame,(left,top),(right,bottom),(0,255,0),2)
            cv2.putText(combinFrame,name,(left,top-6),font,.75,(0,255,0),2)
        
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