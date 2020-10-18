import os
from PIL import Image #PythonImageLibrary for working with training images
import numpy as np
import cv2
import pickle #to save labels for use in face_recognition.py

import tensorflow as tf #for advanced classification
import tensorflow_hub as hub
from tensorflow.keras import layers

#"walk" through files, looking for .jpg, .jpeg, or .png and add them to list of training objects

#image directory
img_dir = os.path.abspath('C:\\Users\\aarya\\Documents\\Python\\facial_recognition_opencv\\faces')

#cascade classifier to detect region of interest later
face_cascade = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")

#lists for training and labels
x_train = []
y_labels = []

#different data will be saved for tensorflow
tf_x_train = []

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
            color_array_img = np.array(Image.open(path), 'uint8') #save the colored version of the previous two lines

            #find region of interest in each img to focus training
            faces = face_cascade.detectMultiScale(array_img, scaleFactor=1.2, minNeighbors=3)
            #scaleFactor—zoom out image to speed up process of finding a face; minNeighbors — higher=quality over quantity

            for (x,y,w,h) in faces: #because faces returns coordinates of rectangle where face is
                #print(f"{count}---{y_id}") #count variable is used to test whether cascade detects right number of faces in all images
                roi = array_img[y:y+h, x:x+w]
                color_roi = color_array_img[y:y+h, x:x+w]

                x_train.append(roi) #add the region of interest to training data
                tf_x_train.append(color_roi) #the tensorflow model needs color
                y_labels.append(y_id) #add the corresponding y_id to the training labels

            #count-=1
#previous code isolates x_train and y_labels


#print(x_train)
#print(tf_x_train)
#print(y_labels)


#uncomment to train the LBPH model
#pickle labels (save in .pkl file) for use in face_recognition.py
with open("labels.pkl", 'wb') as f:
    pickle.dump(label_ids, f)

# #create facial recognizer (try LBPH)
# recognizer = cv2.face.LBPHFaceRecognizer_create()

# #train classifier (make sure both are in numpy array form)
# recognizer.train(x_train, np.array(y_labels))
# recognizer.save("trainer.yml") #save as YAML file





#tensorflow classification (due to the small amount of training data, I will use a pre-trained classifier from TensorFlow Hub)

#preprocess data for use by tensorflow classifier (resize and normalize)
def preprocess(arr):
    resized = np.array([tf.image.resize(img, (224, 224)) for img in arr])
    return resized / 255.

tf_train = preprocess(tf_x_train)
tf_labels = y_labels

#the classifier that will be used is ResNet
#use feature_vector version, not classification; feature_vector has last layer removed so it's customizable

URL = 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4' #url of the model
resnet = hub.KerasLayer(URL, trainable=False) #don't train existing parameters

#build the Sequential model (put in the ResNet feature vector, then add my output layer with appropriate number of neurons)
model = tf.keras.Sequential(layers=[
    resnet,
    tf.keras.layers.Dense(current_id, activation='softmax') #softmax activation for classification (produces prob. dist.)
])

model.build([None, 224, 224, 3]) #build the model (batch size of none; input size of 224x224 and 3 color channels RGB)

print(model.summary()) #if I want to view the model's parameters (23,564,800 trainable; 6,147 untrainable)

#compile the model
model.compile(
    optimizer='adam', #adam optimizer is industry-standard; best optimizer
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #standard loss used for classification (ResNet doc.)
    metrics=['accuracy']
)

#train the model
EPOCHS = 15
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2, restore_best_weights=True) #stop the model from overfitting

history = model.fit(
    x=np.asarray(tf_train).astype(np.float32),
    y=np.asarray(tf_labels).astype(np.float32),
    epochs=EPOCHS,
    verbose=1,
    shuffle=True,
    callbacks=[early_stop]
)

#save the model for use in face_recognition.py
model.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tf_saved_model'))