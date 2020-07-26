import os
from PIL import Image #PythonImageLibrary for working with training images
import numpy as np
import cv2
import pickle #to save labels for use in face_recognition.py

#"walk" through files, looking for .jpg, .jpeg, or .png and add them to list of training objects

#gives me the directory of where this code is (same directory as pics)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#image directory
img_dir = os.path.join(BASE_DIR, "faces")

#cascade classifier to detect region of interest later
face_cascade = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")

#lists for training and labels
x_train = []
y_labels = []

#empty dictionary to assign labels to unique number for classification
#label_ids: keys = folder name (aryan, ansh), values = id (0, 1)
label_ids = {}
current_id = 0

#nested for loops to get the individual image files (root-'faces'; dirs-'ansh','aryan'; files-images)
for root, dirs, files in os.walk(img_dir):
    # count = 7 #count variable used to check whether cascade is detecting the right number of faces in all images
    for file in files:
        img_suffixes = ("jpg", "jpeg", "png")
        if file.endswith(img_suffixes):
            path = os.path.join(root, file)

            #file names are 1, 2, 3, etc. The actual labels for images should be the directory name
            label = os.path.basename(root).replace(" ", "_").title() #replace, title keeps naming conventions the same

            #add label to dictionary with corresponding numerical id
            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1

            y_id = label_ids[label] #to make it easy to append to y_labels list

            #convert to numpy array, grayscale
            pil_img = Image.open(path).convert("L") #make image grayscale with "L" keyword
            array_img = np.array(pil_img, "uint8") #uint8 makes each value in array 8-bit unsigned integer (0-255)

            #find region of interest in each img to focus training
            faces = face_cascade.detectMultiScale(array_img, scaleFactor=1.2, minNeighbors=3)

            for (x,y,w,h) in faces: #because faces returns coordinates of rectangle where face is
                #print(f"{count}---{y_id}") #count variable is used to test whether cascade detects right number of faces in all images
                roi = array_img[y:y+h, x:x+w]
                x_train.append(roi) #add the region of interest to training data
                y_labels.append(y_id) #add the corresponding y_id to the training labels

            #count-=1
#previous code isolates x_train and y_labels
#go from here for scikit-learn possibility

#print(x_train)
#print(y_labels)


#pickle labels (save in .pkl file) for use in face_recognition.py
with open("labels.pkl", 'wb') as f:
    pickle.dump(label_ids, f)

#create facial recognizer (try LBPH)
recognizer = cv2.face.LBPHFaceRecognizer_create()

#train classifier (make sure both are in numpy array form)
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml") #save as YAML file
