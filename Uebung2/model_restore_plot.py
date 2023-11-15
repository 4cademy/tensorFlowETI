# Import the required packages

from __future__ import absolute_import, division, print_function, unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=INFO, 1=WARNING, 2=ERROR, 3=CRITICAL
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
from PIL import Image
import os
from tensorflow.python import pywrap_tensorflow

# Download the dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Class names for plotting
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Scale the data
train_images = train_images / 255.0
test_images = test_images / 255.0


model1 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu', kernel_initializer='GlorotUniform'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model. We will define the loss function, an optimizer and the a metric for observing the training accuracyself.
model1.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print('new model')

img = test_images[3]
print(colored((img.shape), 'green'))

img = (np.expand_dims(img,0))
print(colored((img.shape), 'green'))

predictions_single = model1.predict(img)

print(colored(np.argmax(predictions_single[0]), 'green'))
print(colored('test label', 'red'), colored(test_labels[3], 'red'))

print('after loading')
model1.load_weights('./checkpoints/my_checkpoint')
predictions_single = model1.predict(img)

print(colored(np.argmax(predictions_single[0]), 'green'))
print(colored('test label', 'red'), colored(test_labels[3], 'red'))

model_dir = './checkpoints'
checkpoint_path = os.path.join(model_dir, "model.ckpt")
# reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
# var_to_shape_map = reader.get_variable_to_shape_map()
# var_to_shape_map = checkpoint.restore(checkpoint_path)
reader = tf.train.load_checkpoint('./checkpoints/')
shape_from_key = reader.get_variable_to_shape_map()
dtype_from_key = reader.get_variable_to_dtype_map()

print(sorted(shape_from_key.keys()))
key = 'layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE'
print("Shape:", shape_from_key[key])
print("Dtype:", dtype_from_key[key].name)

################################

values = reader.get_tensor(key)
print(type(values))
print(values.shape)
print(values)
value_plt = values.flatten()
print('maximum values is :', max(value_plt))
print('minimum values is :', min(value_plt))

plt.figure()
plt.hist(value_plt)
plt.show()
