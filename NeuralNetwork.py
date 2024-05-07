#importing libraries
from msilib import Binary
import string
from token import STRING
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import keras
from matplotlib import _cm
from matplotlib import _cm_listed
from matplotlib import colormaps
from PIL import Image
from pathlib import Path

# total amount of user files being looked at. ALL FILES BEING LOOKED AT SHOULD BE IN THIS DIRECTORY
# the naming scheme for these files goes like this "testImage" + number starting from 1 increasing from there + ".png"
# so for example it would go "testImage1.png", "testImage2.png", etc.
totalFiles = 3

# getting the mnist numbers dataset. it contains 60000 training images and 10000 test images.
numbers_mnist = keras.datasets.mnist

# putting the training and testing data into variables
(trainingImages, trainingImageLabels), (testingImages, testingImageLabels) = numbers_mnist.load_data()

# list of all the classifications names
classNames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# showing off a training image
plt.figure()
plt.imshow(trainingImages[1])
plt.show()


# converting all of the image data to be values 0 to 1. This is so it can be put into input nodes of the neural network.
# The original values are from 0 to 255 so this should allow everything to be valid.

trainingImages = trainingImages / 255.0

testingImages = testingImages / 255.0

# this displays some of the training images along with their labels
plt.figure(figsize=(8, 8))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(trainingImages[i], cmap=plt.cm.get_cmap("binary"))
    plt.xlabel(classNames[trainingImageLabels[i]])
plt.show()


# making the neural network. The flatten layer is the input nodes, the middle layer is just a regular hidden layer, and the last layer is the output nodes.
# it is using a relu activation function which is: max(0, number)

network = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(150, activation='relu'),
    keras.layers.Dense(10)
])

# compiling the network. I'm unsure exactly what the optimizer does, but you need this specific loss function when you have multiple classifications
# the metric we are looking at is accuracy of the guess

network.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# training the network. We are going through all of the training data with 10 epochs
network.fit(trainingImages, trainingImageLabels, epochs=10)

# this is generating the bars in the terminal during the training session. Verbose 2 is used in noninteractive inputs.
testLoss, testAccuracy = network.evaluate(testingImages,  testingImageLabels, verbose="2")

# adding the softmax layer here changes the output to be displayed in probabilities, allowing for showing the correct classifier to be easier.
probabilityNetwork = keras.Sequential([network,
                                         keras.layers.Softmax()])


# running the network with its test data to see its effectiveness.
predictions = probabilityNetwork.predict(testingImages)


# showing a grid of predictions along with their images
plt.figure(figsize=(10, 10))
for i in range(18):
    plt.subplot(3,6,i+1)

    if(i % 2 == 0):
      plt.grid(False)
      plt.xticks([])
      plt.yticks([])
      plt.imshow(testingImages[i], cmap=plt.cm.get_cmap("binary"))
    else:
      plt.grid(False)
      plt.xticks(range(10))
      plt.yticks([])
      plt.bar(range(10), predictions[i-1])
      plt.ylim([0, 1])

plt.tight_layout()
plt.show()



# This for loop runs a prediction on all of the user given photos.
for number in range(1, totalFiles+1):

  # getting all images one by one
  script_dir = Path(__file__).resolve().parent
  fileName = "testImage" + str(number) + ".png"
  filePath = script_dir / fileName

  #print("image path :: ")
  #print(file_path)

  # opening then converting image and resizing it if needed
  image = Image.open(filePath).convert('L')
  image = image.resize((28, 28))



  # Convert the image to a numpy array and convert the image to the same values as training and testing images
  image_array = np.array(image) / 255.0  

  # Add the image to a batch where it's the only member.
  image_array = np.expand_dims(image_array, axis=0)  

  # you have to invert the values of the array or they are all the opposite of the testing and training data
  image_array = abs(1-image_array)


  # making the prediction
  prediction = probabilityNetwork.predict(image_array)

  print("Predicted class:", np.argmax(prediction))


  # plotting the predictions image along with the guess
  plt.figure()
  plt.subplot(1,2,1)
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(image, cmap=plt.cm.get_cmap("binary"))


  plt.subplot(1,2,2)
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  plt.bar(range(10), prediction[0])
  plt.ylim([0, 1])
  plt.show()