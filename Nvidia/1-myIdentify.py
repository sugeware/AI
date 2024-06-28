import cv2
import numpy as np
import jetson.inference
import jetson.utils

print(jetson.inference.VERSION)

cam = jetson.utils.gstCamera(640, 480, '/dev/video0')
disp = jetson.utils.glDisplay()
net = jetson.inference.imageNet('googlenet')
font = jetson.utils.cudaFont()

while disp.IsOpen():
    frame, width, height = cam.CaptureRGBA()
    # print(width,height)
    # frame2,width,height = cam.Capture()
    # print('frame1:'+str(frame1))
    # print('frame2'+str(frame2))
    classID, confident= net.Classify(frame, width, height)
    
    item = net.GetClassDesc(classID)
    font.OverlayText(frame, width, height, item, 5, 5,font.Magenta,font.Blue)
    # cv2.rectangle(frame,)
    disp.RenderOnce(frame,width,height)
