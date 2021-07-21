# Facial Recognition and Mask Detection with OpenCV and Tensorflow
This is a Python facial recognition program that can identify and label faces in the computer's live webcam footage, as well as detect whether a facemask is being worn correctly. 

## Classification Methods

### Face and Facemask Detection
4 total Haar Cascades were used to detect faces:
1. Frontal Face Alt2 (detects faces for the purpose of recognition)
2. Frontal Face Default (detects faces for the purpose of facemask detection)
3. MCS Mouth (detects mouths in the frame)
4. MCS Nose (detects noses in the frame)

A mask is "detected" if the cascades detect a face, but do not detect a mouth or a nose in the image.

### LBPH Face Recognizer
Face recognizer built into the OpenCV library.

### TensorFlow Transfer Learning Model (ResNet 50)
A Keras Sequential model with two layers (~24,000,000 total parameters, depending on number of classes).
1. the Inception image classification model (trained on the ImageNet dataset) with last layer removed
2. a Dense layer with one neuron for each potential face

It performs *much* better than the LBPH recognizer.

## Functions of the 3 .py Files
* `get_face_data.py` saves images of a user's face
* `face_train.py` trains the LBPH face recognizer and TensorFlow model on those images and labels
* `face_recognition.py` shows live webcam footage with a rectangle around the faces in an image and the corresponding name (label). It also shows whether the subject is wearing a facemask.

## Running the Program
Run the .py files in the order they are presented above. `face_recognition.py` will show live webcam footage with a rectangle around the faces in the shot and the corresponding name, as well as one of three messages telling the user whether a facemask is detected or not (detected, improperly worn, not detected). 

The LBPH recognizer's classification is shown in green above the box (this is optional; uncomment the appropriate lines in `face_recognition.py` and `face_train.py` to see this). The TensorFlow model's classification is shown in orange below the box. The facemask detection message is shown at the bottom of the frame in either green, yellow, or red. 

## Libraries Used
* OpenCV
* TensorFlow
* os
* numpy
* PIL
* pickle
* matplotlib.pyplot
* seaborn

#### Notes:
1. The actual images in the faces directory have been removed for privacy
2. `variables.data-00000-of-00001` (where the weights of the neural network are stored) has been omitted due to sheer size