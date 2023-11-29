import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as pl
import math

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print(tf.version.VERSION)

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels
test_labels = test_labels

train_images = train_images.reshape(-1, 28 * 28) / 255.0
test_images = test_images.reshape(-1, 28 * 28) / 255.0

print(train_images.shape, test_images.shape)  # (1000, 784) (1000, 784)


# Define a simple sequential model
def create_model():
    model = tf.keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        # keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    return model


# to quantize a tensor
def quantize(values, bits):
    range = 2 ** (bits - 1)
    maxVal = tf.reduce_max(tf.abs(values))
    step = (maxVal / range)
    step = tf.pow(2.0, tf.math.ceil(tf.math.log(step) / math.log(2)));
    qValues = tf.round(values / tf.cast(step, tf.float32))
    qValues = step * tf.clip_by_value(qValues, -range, range - 1)
    return qValues


# Create a basic model instance
model = create_model()

# Display the model's architecture
model.summary()

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Train the model with the new callback
model.fit(train_images,
          train_labels,
          epochs=5,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])  # Pass callback to training

# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook)
# are in place to discourage outdated usage, and can be ignored.

# Create a basic model instance
model = create_model()

# Evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

# Loads the weights # to turn off the warnings from tensorflow v2, use ---> .expect_partial()
model.load_weights(checkpoint_path).expect_partial()

# shape of the weights
print("shape of the weights")
for w in model.get_weights():
    print(w.shape)

'''for layer in model.layers:
  print(layer.name)
  print("Weights")
  print("Shape: ",layer.get_weights()[0].shape,'\n',layer.get_weights()[0])
  print("Bias")
  print("Shape: ",layer.get_weights()[1].shape,'\n',layer.get_weights()[1],'\n')'''

loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Floating point Restored model, accuracy: {:5.2f}%".format(100 * acc))

quant_bits = [1, 2, 4, 6, 8, 16]
for w_bits in quant_bits:
    model = create_model()
    model.load_weights(checkpoint_path).expect_partial()
    for layer in model.layers:
        # layer.set_weights([layer.get_weights()[0], layer.get_weights()[1]])
        layer.set_weights([quantize(layer.get_weights()[0], w_bits), quantize(layer.get_weights()[1], w_bits)])

    # Re-evaluate the model
    loss, acc = model.evaluate(test_images, test_labels, verbose=2)
    print(w_bits, "bit quantized restored model, accuracy: {:5.2f}%".format(100 * acc))
