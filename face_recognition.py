import cv2
import pickle
import os
import numpy as np
import tensorflow as tf

#preprocessing function used for tensorflow
def preprocess(img):
    resized = tf.image.resize(img, (224, 224))
    return resized / 255.

#get labels in .pkl file saved in face_train.py
labels = {}
with open("labels.pkl", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()} #invert the dictionary so that key is value (0 or 1) and value is key (aryan, ansh)
#labels: keys = id_ (0,1); values = aryan or ansh

#define the cascade that finds faces
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

#define cascades for finding noses and mouths (for mask recognition)
face_detect = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml') #because alt2 won't detect masked faces
nose_detect = cv2.CascadeClassifier('cascades/data/haarcascade_mcs_nose.xml')
mouth_detect = cv2.CascadeClassifier('cascades/data/haarcascade_mcs_mouth.xml')

#initialize recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

#load training
recognizer.read("trainer.yml")

#load the saved TensorFlow model
tf_model = tf.keras.models.load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tf_saved_model'))

cap = cv2.VideoCapture(0)

while True:
    #capture frame-by-frame
    ret, frame = cap.read()

    #convert frames to grayscale so classifier works (see documentation)
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #flip horizontally (experimentation)
    flipped = cv2.flip(grayscale, 1)

    #find faces in frame (detectMultiScale returns list of rectangles where face found)
    faces = face_cascade.detectMultiScale(grayscale, scaleFactor=1.5, minNeighbors=4)

    #iterate over rectangles returned by faces
    for (x, y, w, h) in faces:

        #region of interest (isolate the rectangle in the image with my face)
        roi_color = frame[y:y+h, x:x+w]
        roi_gray = grayscale[y:y+h, x:x+w]

        #predict the face's ID (id_) and (loss): the lower the better   --> try tensorflow
        id_, loss = recognizer.predict(roi_gray)
        tf_id = np.argmax(tf_model.predict(np.array([preprocess(roi_color)])))

        font = cv2.FONT_HERSHEY_COMPLEX
        name = labels[id_]
        color = (0, 255, 0) #BGR
        stroke = 2

        #uncomment for LBPH classification for comparison
        #cv2.putText(frame, name, (x, y+h+20), font, 1, color, thickness=stroke, lineType=cv2.LINE_AA)

        tf_font = cv2.FONT_HERSHEY_COMPLEX
        tf_name = labels[tf_id]
        tf_color = (0, 145, 255) #BGR
        tf_stroke = 2
        cv2.putText(frame, tf_name, (x,y), tf_font, 1, tf_color, thickness=tf_stroke, lineType=cv2.LINE_AA)

        #save the region of interest as an image
        #cv2.imwrite("color_face.png", roi_color)

        #draw a rectangle surrounding face
        rec_color = (255, 80, 0) #BGR (a nice blue)
        rec_stroke = 2 #thickness
        cv2.rectangle(frame, (x, y), (x+w, y+h), rec_color, rec_stroke)

    #MASK DETECTION SECTION
    faces = face_detect.detectMultiScale(grayscale, scaleFactor=None, minNeighbors=None)

    face_list = []


    for (x,y,w,h) in faces:
        if w*h > 60000: #test for detected 'faces' that are too small to be actual faces
            face_list.append(w*h)

    #choose the face with the biggest region of interest area
    if len(face_list) > 0:
        face_index = np.argmax(face_list)
        face = faces[face_index]
        
        x = face[0]
        y = face[1]
        w = face[2]
        h = face[3]
        roi = grayscale[y:y+h, x:x+w] #roi has different coordinates than frame

        # uncomment to see another rectangle around the face
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (255,105,0), 2, cv2.LINE_AA) #a nice blue

        detected_mouths = mouth_detect.detectMultiScale(roi, scaleFactor=None, minNeighbors=4)
        mouth_list = []

        for (x,y,w,h) in detected_mouths:
            if w*h > 4200 and y > (2 * (face[3] // 3)): #eliminate "mouths" that are too small and not in the bottom 1/3 of the face
                mouth_list.append(w*h)

        #choose the "mouth" with the biggest area (eyes tend to be classified as mouths)
        # uncomment the code block if you want a rectangle drawn around the mouth in the video feed
        # if len(mouth_list) > 0:
            
        #     mouth_index = np.argmax(mouth_list)
        #     mouth = detected_mouths[mouth_index]

        #     mouthx = face[0] + mouth[0]
        #     mouthy = face[1] + mouth[1]

        #     cv2.rectangle(frame, (mouthx, mouthy), (mouthx+mouth[2], mouthy+mouth[3]), (0, 255, 0), 2, cv2.LINE_AA) #green

        detected_noses = nose_detect.detectMultiScale(roi, scaleFactor=None, minNeighbors=4)

        nose_list = []

        for (x,y,w,h) in detected_noses:
            if w*h > 5300 and ((face[3] // 3)) < y < (2 * (face[3] // 3)): #eliminate "noses" that are too small and not in the middle 1/3 of the face
                nose_list.append(w*h)

        #choose the nose with the biggest area
        # uncomment the code block if you want a rectangle drawn around the nose in the video feed
        # if len(nose_list) > 0:
            
        #     nose_index = np.argmax(nose_list)
        #     nose = detected_noses[nose_index]

        #     nosex = face[0] + nose[0]
        #     nosey = face[1] + nose[1]

        #     cv2.rectangle(frame, (nosex, nosey), (nosex+nose[2], nosey+nose[3]), (0, 0, 255), 2, cv2.LINE_AA) #red


        #defines the text parameters
        mask_font = cv2.FONT_HERSHEY_TRIPLEX
        mask_name = ''
        mask_color = (0, 0, 255) #BGR
        mask_stroke = 2

        if len(mouth_list) > 0 and len(nose_list) > 0: #both mouth and nose detected
            mask_color = (0, 0, 255) # red
            mask_name = "Mask Not Detected"
        elif len(mouth_list) == 0 and len(nose_list) == 0: #neither mouth nor nose detected
            mask_color = (0, 255, 0) #make the text color green
            mask_name = "Mask Detected"
        else: # mouth or nose detected, but not both
            mask_color = (0, 255, 255) # yellow
            mask_name = "Incorrectly Worn Mask Detected"

        #get rectangle boundary of text
        text_size = cv2.getTextSize(mask_name, mask_font, 1, mask_stroke)[0]

        #frame.shape returns a tuple of (height, width)
        #determine coords of text based on its size and the size of the frame (bottom middle)
        mask_x = (frame.shape[1] - text_size[0]) // 2
        mask_y = frame.shape[0]

        cv2.putText(frame, mask_name, (mask_x, mask_y), mask_font, 1, mask_color, thickness=mask_stroke, lineType=cv2.LINE_AA)


    #display resulting frame
    cv2.imshow('Live Webcam Feed', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

#when everything is done, release the capture (end)
cap.release()
cv2.destroyAllWindows()
