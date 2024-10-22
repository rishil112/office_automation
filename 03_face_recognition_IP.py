''''
Real Time Face Recogition
	==> Each face stored on dataset/ dir, should have a unique numeric integer ID as 1, 2, 3, etc                       
	==> LBPH computed model (trained faces) should be on trainer/ dir
'''

import cv2

recognizer = cv2.face.LBPHFaceRecognizer()
recognizer.read('trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

# names related to ids: example ==> Name: id=1,  etc
names = ['None', 'Sachin', 'Rishil', 'Vaibhav', 'Ritik', 'Neeraj','Dipesh','Shivam','Deepali','Sukant']

# Initialize and start realtime video capture
#cam = cv2.VideoCapture(0) # For CSI Camera

def recognize_faces(cam_id):
    cam = cv2.VideoCapture(f'rtsp://admin:denmarkOPD8188@103.189.220.219:554/ISAPI/Streaming/Channels/{cam_id}')
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height
    cam.set(cv2.CAP_PROP_FPS, 10)

    # Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:

        ret, img = cam.read()
        img = cv2.flip(img, -1)  # Flip vertically

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            # Check if confidence is less them 100 ==> "0" is perfect match
            if confidence < 100:
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        if cam_id == 501:
            img = cv2.resize((1080, 720), img)
        cv2.imshow('camera', img)

        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break

    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()

recognize_faces(501)
recognize_faces(101)
