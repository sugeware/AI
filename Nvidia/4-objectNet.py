import jetson.inference
import jetson.utils
import cv2
import numpy as np 
import time

width = 640
height = 480
dispW = width
dispH = height

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
net = jetson.inference.detectNet('ssd-mobilenet-v2',threshold = .5)

timeStamp = time.time()
fpsFilter = 0
font = cv2.FONT_HERSHEY_COMPLEX

while True:
    ret,frame = cam.read()
    
    if not ret:
        print('Fail to open camera!!!')
        break
    
    width = frame.shape[0]
    height = frame.shape[1]
    
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA).astype(np.float32)
    img = jetson.utils.cudaFromNumpy(img)
    
    detections = net.Detect(img,width,height)
    for detect in detections:
        print(detect)
        id = detect.ClassID
        top = int(detect.Top)
        bottom = int(detect.Bottom) 
        left = int(detect.Left) 
        right = int(detect.Right) 
        item = net.GetClassDesc(id)
        if item == "person":
            cv2.rectangle(frame,(left,top),(right,bottom),(255,0,0),1)
            cv2.putText(frame,item,(left+1,top),font,.75,(0,0,255),2)
            
    dt = time.time()-timeStamp
    fps = 1/dt
    fpsFilter = .95*fpsFilter + .05*fps    
    timeStamp = time.time()
            
    cv2.imshow('video',frame)
    cv2.setWindowTitle('video',f'fps:{round(fpsFilter,1)}')
    cv2.moveWindow('detect',0,0)    

    if cv2.waitKey(1) == ord('q'):
        break
    
cam.release()
cv2.destroyAllWindows()

