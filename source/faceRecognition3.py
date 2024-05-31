import face_recognition
import os
import cv2
print(face_recognition.__version__)

Encodings = []
Names = []

# -----------Trainning the face recognition models------------
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

# -----------Face recognition--------------
font = cv2.FONT_HERSHEY_COMPLEX
image_dir = '/home/jetson/Desktop/AI/faceReconimizer/unknown'
for root, dirs, files in os.walk(image_dir):
    for file in files:
        print(root)
        print(file)
        testImagePath = os.path.join(root,file)
        testImage = face_recognition.load_image_file(testImagePath) 
        facePositions = face_recognition.face_locations(testImage)
        allEncodings = face_recognition.face_encodings(testImage,facePositions)
        testImage = cv2.cvtColor(testImage,cv2.COLOR_RGB2BGR)
        
        for (top,right,bottom,left),face_encoding in zip(facePositions,allEncodings):
            name = 'Unkonwn Person'
            matchs = face_recognition.compare_faces(Encodings,face_encoding)
            if True in matchs:
                first_match_index = matchs.index(True)
                name = Names[first_match_index]
            cv2.rectangle(testImage,(left,top),(right,bottom),(0,255,0),2)
            cv2.putText(testImage,name,(left,top-6),font,.75,(0,255,0),2)
        cv2.imshow('picture',testImage)
        cv2.moveWindow('picture',0,0)
        if cv2.waitKey(0)==ord('q'):
                cv2.destroyAllWindows()