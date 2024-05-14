import cv2
import numpy as np
import sys
import adafruit_servokit as servoKit

# url = 'rtsp://admin:su619865277@192.168.110.100:554/h264/ch01/main/av_stream'
cap = cv2.VideoCapture(0,cv2.CAP_V4L2)
# cap = cv2.VideoCapture(url)

while True:
    ret,frame = cap.read()

    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()