import cv2
import numpy as np 
import jetson.inference
import jetson.utils
import time

width = 640
height = 480
dispH = height
dispW = width
# camUrl = 'rtsp://admin:Su619865277@192.168.110.105/ch1-s1/tcp'
# cam = cv2.VideoCapture(camUrl)
cam = cv2.VideoCapture(1,cv2.CAP_V4L2)
# cv2.namedWindow('frame',cv2.WINDOW_AUTOSIZE)
# cv2.resizeWindow('frame',width,height)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, dispW)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, dispH)
font = cv2.FONT_HERSHEY_SIMPLEX

net = jetson.inference.imageNet('googlenet')

fpsFilter = 0
timeStamp = time.time()

while True:
    ret,frame = cam.read()
    # frame = cv2.resize(frame,(width,height))
    
    if ret == False:
        print('Fail to open cam!')
        break
    else:
        frameRGBA = cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA).astype(np.float32)
        frameRGBA = jetson.utils.cudaFromNumpy(frameRGBA)
        classID,confidence = net.Classify(frameRGBA,width,height)
        item = net.GetClassDesc(classID)
        
        dt = time.time()-timeStamp
        fps = 1/dt
        fpsFilter = .95*fpsFilter + .05*fps    
        timeStamp = time.time()
        
        # frame = jetson.utils.cudaToNumpy(frameRGBA,width,height,3)
        # frame = cv2.cvtColor(frame,cv2.COLOR_RGBA2BGR).astype(np.uint8)
        # cv2.rectangle(frame,(0,0),(120,30),(0,255,0),2)
              
        cv2.putText(frame,'fps:'+str(round(fpsFilter,1))+';'+item,(0,30),font,1,(0,255,0),2)
        cv2.imshow('frame',frame)
        cv2.moveWindow('frame',0,0)
        
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
        

        
        
        
        
    