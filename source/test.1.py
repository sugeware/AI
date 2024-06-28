import cv2
import numpy as np 
import jetson.inference
import jetson.utils
import time
from threading import Thread

width = 640
height = 480
dispH = height
dispW = width
camUrl = 'rtsp://admin:Su619865277@192.168.110.105/ch1-s1/tcp'
cam = cv2.VideoCapture(camUrl)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, dispW)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, dispH)
font = cv2.FONT_HERSHEY_SIMPLEX

net = jetson.inference.imageNet('googlenet')

fpsFilter = 0
timeStamp = time.time()

def process_frame(frame):
    global fpsFilter, timeStamp
    
    frameRGBA = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA).astype(np.float32)
    frameCUDA = jetson.utils.cudaFromNumpy(frameRGBA)
    
    classID, confidence = net.Classify(frameCUDA, width, height)
    item = net.GetClassDesc(classID)
    
    dt = time.time() - timeStamp
    fps = 1 / dt
    fpsFilter = 0.95 * fpsFilter + 0.05 * fps
    timeStamp = time.time()
    
    cv2.putText(frame, 'fps: {:.1f}; {}'.format(fpsFilter, item), (0, 30), font, 1, (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    cv2.moveWindow('frame', 0, 0)

while True:
    ret, frame = cam.read()
    
    if not ret:
        print('Fail to open cam!')
        break

    t = Thread(target=process_frame, args=(frame,))
    t.start()

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()