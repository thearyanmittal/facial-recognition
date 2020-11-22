import os
from PIL import Image #PythonImageLibrary for working with training images
import numpy as np
import cv2
import pickle #to save labels for use in face_recognition.py
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

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
            color = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
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
                tf_x_train.append(color[y:y+h,x:x+w])
                #tf_x_train.append(color_roi) #the tensorflow model needs color
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

#create facial recognizer (try LBPH)
recognizer = cv2.face.LBPHFaceRecognizer_create()

#train classifier (make sure both are in numpy array form)
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml") #save as YAML file





#tensorflow classification (due to the small amount of training data, I will use a pre-trained classifier from TensorFlow Hub)

#preprocess data for use by tensorflow classifier (resize and normalize)
IMG_SIZE = 224

def normalize(arr):
    resized = np.array([tf.image.resize(img, (IMG_SIZE, IMG_SIZE)) for img in arr])
    return resized / 255.

tf_x_train = normalize(tf_x_train) #make this augmented
tf_y_train = y_labels

#augment data in various ways to 1. add training data and 2. increase model's ability to generalize
generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=45, 
                                                            brightness_range=(0.5,1.5), #this is the main one 
                                                            horizontal_flip=True)


# use created augmentation generator to flow data from the image directory
tf_train = []
tf_labels = []
count = 0
increase_factor = 2 #change this according to how many training images you want/have
start_time = datetime.datetime.now()

for x,y in generator.flow(tf_x_train, y_labels, batch_size=1):
    tf_train.append(x)
    tf_labels.append(y)
    # print((increase_factor*len(pre_tf_train)) - count) #if you want the program to count down the images
    count += 1
    if count == (int(increase_factor * len(tf_x_train))): #when the set of training data has doubled, that's enough 
        break

end_time = datetime.datetime.now()
print(f"Total Time to Process Training Data: {end_time - start_time}")

#reshape it into something the model accepts (remove batch size)
tf_minus_batch = []
for batch in tf_train:
    for img in batch:
        tf_minus_batch.append(img)

tf_train = np.array(tf_minus_batch)

#view some of the augmented images
def plot_imgs(arr):
    num_imgs = 4 # I want 4 rows and 4 columns of images
    plt.figure(figsize=(10,10))
    for i in range(num_imgs * num_imgs):
        plt.subplot(num_imgs, num_imgs, i+1)
        plt.imshow(arr[i].astype(np.uint8))


plot_imgs(tf_train)
plt.show()

tf_train = normalize(tf_train) #normalize all the augmented data

#the classifier that will be used is Resnet 50
#use feature_vector version, not classification; feature_vector has last layer removed so it's customizable

URL = 'https://tfhub.dev/tensorflow/resnet_50/feature_vector/1' #url of the model
resnet = hub.KerasLayer(URL, trainable=False) #don't train existing parameters

#build the Sequential model (put in the ResNet feature vector, then add my output layer with appropriate number of neurons)
model = tf.keras.Sequential(layers=[
    resnet,
    tf.keras.layers.Dropout(0.5), #added a dropout layer to increase model's generalization ability and future accuracy
    tf.keras.layers.Dense(current_id, activation='softmax'), #softmax activation for classification (produces prob. dist.)
])

model.build([None, IMG_SIZE, IMG_SIZE, 3]) #build the model (batch size of 1; expects input size of 299x299 and 3 color channels RGB)

print(model.summary()) #if I want to view the model's parameters (23,569,348 untrainable; 8,196 trainable for 4 classes)

#compile the model
model.compile(
    optimizer='adam', #adam optimizer is industry-standard; best optimizer
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #standard loss used for classification (ResNet doc.)
    metrics=['accuracy']
)

#train the model
EPOCHS = 15
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True) #stop the model from overfitting

history = model.fit(
    x=np.asarray(tf_train),
    y=np.asarray(tf_labels),
    epochs=EPOCHS,
    verbose=1,
    shuffle=True,
    callbacks=[early_stop]
)

#save the model for use in face_recognition.py
model.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tf_saved_model'))

#create graphs of training accuracy and loss
plt.figure(figsize=(10,10))
plt.suptitle("Loss and Accuracy over Training")

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'r')
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'b')
plt.title('Accuracy')

plt.show()