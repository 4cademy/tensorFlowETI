import os

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as pl

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print(tf.version.VERSION)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

def flatten(l):
    return [item for sublist in l for item in sublist]


# Define a simple sequential model
def create_model():
  model = tf.keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,), use_bias =False),
    #keras.layers.Dropout(0.2),
    keras.layers.Dense(10, use_bias =False)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

  return model

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
          epochs=20,
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


# Loads the weights
model.load_weights(checkpoint_path)

#shape of the weights
print("shape of the weights")
for w in model.get_weights():
    print(w.shape) 


# Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


#saving histograms of different layers of model
for i in range(len(model.get_weights())):
    print("layer", i, "number of paramerters", len(flatten(model.layers[i].get_weights()[0])))
    fig = pl.hist(flatten(model.layers[i].get_weights()[0]))
    pl.xlabel("value")
    pl.ylabel("Frequency")
    name = "layer"+str(i)+".png"
    pl.savefig(name)
    pl.savefig("training_1/"+name)
    pl.clf()


