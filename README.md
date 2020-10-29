# Facial Recognition with OpenCV and Tensorflow
This is a Python facial recognition program that can identify and label faces in the computer's live webcam footage. 

## Classification Methods

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
* `face_recognition.py` shows live webcam footage with a rectangle around the faces in an image and the corresponding name (label).

## Running the Program
Run the .py files in the order they are presented above. `face_recognition.py` will show live webcam footage with a rectangle around the faces in the shot and the corresponding name. 

The LBPH recognizer's classification is shown in green above the box (this is optional; uncomment the appropriate lines in `face_recognition.py` and `face_train.py` to see this). The TensorFlow model's classification is shown in orange below the box.

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
The actual images in the faces directory have been removed for privacy
`variables.data-00000-of-00001` (where the weights of the neural network are stored) has been omitted due to sheer size