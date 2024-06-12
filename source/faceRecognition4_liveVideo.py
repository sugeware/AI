import face_recognition
import os
import cv2
import pickle
import time
print(face_recognition.__version__)

train = False

# -----------Trainning the face recognition models------------
if train == True:
    print('Begain to train...')
    Encodings = []
    Names = []
    image_dir = '/home/jetson/Desktop/AI/faceReconimizer/known'
    for root ,dirs, files in os.walk(image_dir):
        print(files)
        for file in files:
            path = os.path.join(root,file) # Build file path
            print(path)
            name = os.path.splitext(file)[0] # Extract the body ofa file name from a file(without the extension)
            print(name)
            person = face_recognition.load_image_file(path) #  Load the image 
            Encoding = face_recognition.face_encodings(person)[0] # The feature encoding of a face is tetracted from a face image
            Encodings.append(Encoding)
            Names.append(name)
    print(Names)

    with open('/home/jetson/Desktop/AI/models/train2.pkl','wb') as f:
        pickle.dump(Names,f)
        pickle.dump(Encodings,f)
   
Encodings = []
Names = []

with open('/home/jetson/Desktop/AI/models/train2.pkl','rb') as f:
    Names = pickle.load(f)
    Encodings = pickle.load(f)

# -----------Face recognition--------------
font = cv2.FONT_HERSHEY_COMPLEX
cam_url = 'rtsp://admin:Su619865277@192.168.110.166/ch1-s1/tcp'
# cam = cv2.VideoCapture(cam_url)
cam = cv2.VideoCapture(0,cv2.CAP_V4L2)
cv2.namedWindow('picture',cv2.WINDOW_NORMAL)
cv2.resizeWindow('picture',640,480)


while True:
    _,frame = cam.read()
    frameSmall = cv2.resize(frame,(0,0),fx=.25,fy=.25)
    frameRGB = cv2.cvtColor(frameSmall,cv2.COLOR_BGR2RGB)
    facePositions = face_recognition.face_locations(frameRGB)
    # facePositions = face_recognition.face_locations(frameRGB,number_of_times_to_upsample=1,model="cnn")
    allEncodings = face_recognition.face_encodings(frameRGB,facePositions)
    
    for (top,right,bottom,left),face_encoding in zip(facePositions,allEncodings):
        name = 'Unkonwn Person'
        matchs = face_recognition.compare_faces(Encodings,face_encoding)
        if True in matchs:
            first_match_index = matchs.index(True)
            name = Names[first_match_index]
        top = top*4
        right = right*4
        bottom = bottom*4
        left = left*4    
        cv2.rectangle(frame,(left,top),(right,bottom),(0,255,0),2)
        cv2.putText(frame,name,(left,top-6),font,.75,(0,255,0),2)
    
    cv2.imshow('picture',frame)
    cv2.moveWindow('picture',0,0)
    if cv2.waitKey(1)==ord('q'):
        break
    
               
cam.release()
cv2.destroyAllWindows()