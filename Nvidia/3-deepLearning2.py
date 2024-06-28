import jetson.inference
import jetson.utils
import time
import cv2
import numpy as np

width = 1080
height = 720
dispW = width
dispH = height

cam = jetson.utils.gstCamera(width,height,'/dev/video0')

net = jetson.inference.imageNet('googlenet')

timeStamp = time.time()
fpsFilter = 0
font = cv2.FONT_HERSHEY_COMPLEX

while True:
    frame, width, height = cam.CaptureRGBA(zeroCopy = 1)
    classID,confidence = net.Classify(frame,width,height)
    item = net.GetClassDesc(classID)
    
    frame = jetson.utils.cudaToNumpy(frame).astype(np.uint8)
    frame = cv2.cvtColor(frame,cv2.COLOR_RGBA2BGR)
    
    dt = time.time()-timeStamp
    fps = 1/dt
    fpsFilter = .95*fpsFilter + .05*fps    
    timeStamp = time.time()
    
    # frame = jetson.utils.cudaToNumpy(frameRGBA,width,height,3)
    # frame = cv2.cvtColor(frame,cv2.COLOR_RGBA2BGR).astype(np.uint8)
    # cv2.rectangle(frame,(0,0),(120,30),(0,255,0),2)
            
    cv2.putText(frame,'fps:'+str(round(fpsFilter,1))+';'+item,(0,30),font,.75,(0,255,0),2)
    cv2.imshow('frame',frame)
    cv2.moveWindow('frame',0,0)
        
    if cv2.waitKey(1) == ord('q'):
        break

cam.Close()
cv2.destroyAllWindows()