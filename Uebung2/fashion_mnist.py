# Import the required packages

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
from PIL import Image

# Download the dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Class names for plotting
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Explore the data. Explore the shapes of training and validation dataset
# Not required for building the model.
print (colored(train_images.shape, 'green'))
print(colored(len(train_labels), 'green'))
print(colored(train_labels, 'green'))

# Preprocess the data. We will Normalize the data to the range of 0 to 1. This helps in training the networkself.
# But first print a sample image and look for its pixel values.
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# Scale the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Explore the scaled data by printing few images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Build the model

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu', kernel_initializer='GlorotUniform'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model. We will define the loss function, an optimizer and the a metric for observing the training accuracyself.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with the training data. We will be using the fit function for this purpose
model.fit(train_images, train_labels, epochs=2, batch_size=1, validation_split=0.1)

# Evaluate the accuracy
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)


print('\nTest accuracy:', test_acc)
# Make predictions about all test images and a single test image
# First all images
predictions = model.predict(test_images)
print(colored((predictions[0]), 'green'))
# The maximum probability
print(colored(np.argmax(predictions[0]), 'green'))
# Check if it is a correct prediction by printing the first label in the test dataset
print(colored(test_labels[0], 'red'))

# Now prediction for a single image
# tf.keras models are optimized to make predictions on a batch. So even for a single image, you need to add it to a list:

# Grab an image from the test dataset.
img = test_images[1]
print(colored((img.shape), 'green'))
# add to a list
# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))
print(colored((img.shape), 'green'))

predictions_single = model.predict(img)

print(colored(np.argmax(predictions_single[0]), 'green'))
print(colored(test_labels[1], 'red'))


# img_array= np.array(Image.open('panda.jpg').convert('L').resize((28,28)))
# print(img_array.shape)
# plt.figure()
# plt.imshow(img_array)
# plt.colorbar()
# plt.grid(False)
# plt.show()
#
# img_array = (np.expand_dims(img_array,0))
# print(colored((img_array.shape), 'green'))
#
# predictions_single = model.predict(img_array)
#
# print(colored(np.argmax(predictions_single[0]), 'green'))
# print(colored(test_labels[1], 'red'))
