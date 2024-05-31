import face_recognition as faceRec
import cv2


print('face_recognition_version:'+faceRec.__version__)
print('cv2_version:'+cv2.__version__)

donFace = faceRec.load_image_file('/home/jetson/Desktop/AI/faceReconimizer/known/Donald Trump.jpg')
donEncode = faceRec.face_encodings(donFace)[0]

nancyFace = faceRec.load_image_file('/home/jetson/Desktop/AI/faceReconimizer/known/Nancy Pelosi.jpg')
nancyEncode = faceRec.face_encodings(nancyFace)[0]

Encodings = [donEncode,nancyEncode]
Names = ['The Donald','The Nancy']
# Names = ['The Nancy','The Donald']

font = cv2.FONT_HERSHEY_COMPLEX

testImage = faceRec.load_image_file('/home/jetson/Desktop/AI/faceReconimizer/unknown/u1.jpg')
facePos = faceRec.face_locations(testImage)
allEncodings = faceRec.face_encodings(testImage,facePos)

testImage = cv2.cvtColor(testImage,cv2.COLOR_RGB2BGR)

for (top,right,bottom,left),face_encoding in zip(facePos,allEncodings):
    name = 'Unkoown name'
    matchs = faceRec.compare_faces(Encodings,face_encoding)
    if True in matchs:
        first_match_index = matchs.index(True)
        name = Names[first_match_index]
    cv2.rectangle(testImage,(left,top),(right,bottom),(0,255,0),2)
    cv2.putText(testImage,name,(left,top-6),font,0.25,(0,255,0),1)

cv2.imshow('testImage',testImage)

if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()


