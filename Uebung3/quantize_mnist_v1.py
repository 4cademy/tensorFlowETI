import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=INFO, 1=WARNING, 2=ERROR, 3=CRITICAL
import tensorflow as tf
import numpy as np
import pathlib
import matplotlib.pylab as plt
import matplotlib
from termcolor import colored
# matplotlib.use('GTK3Agg')
# from PIL import Image

####################### Part 1 #:-###################

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images.astype(np.float32)/255.0
test_images = test_images.astype(np.float32)/255.0

# Define the model architecture
model = tf.keras.Sequential([
  tf.keras.layers.InputLayer(input_shape=(28, 28)),
  tf.keras.layers.Reshape(target_shape=(784,)),
  tf.keras.layers.Dense(100),
  tf.keras.layers.Activation('relu'),
  tf.keras.layers.Dense(50),
  tf.keras.layers.Activation('relu'),
  tf.keras.layers.Dense(10),
  tf.keras.layers.Activation('softmax')
])

# model.summary()
# Comment this line if you have not installed pydot and graphviz
#tf.keras.utils.plot_model(model, show_shapes=True, show_dtype=True, show_layer_names=True, to_file='model.png',)


# Train the digit classification model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
model.fit( train_images,  train_labels,  epochs=5,  validation_data=(test_images, test_labels))
result = model.evaluate(test_images, test_labels, verbose=2)
print(result)


####################### Part 2 ####################
############## Four options (a) -- (d)

def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(1000):
    yield [input_value]


## (a) Converting to TfLite without quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

## (b) Converting to TfLite with integer quantization of weights
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model_quant_weights = converter.convert()

## (c) Converting to TfLite with integer quantization of weights and activations
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
tflite_model_quant_weights_acts = converter.convert()

# Check data types of inputs and outputs
interpreter = tf.lite.Interpreter(model_content=tflite_model_quant_weights_acts)
input_type = interpreter.get_input_details()[0]['dtype']
print(colored('input: ', 'green'), colored(input_type, 'green'))
output_type = interpreter.get_output_details()[0]['dtype']
print(colored('output: ', 'green'), colored(output_type, 'green'))


## (d) Converting to TfLite with integer quantization of weights and activations with quantization of inputs and outputs
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model_quant_all = converter.convert()

interpreter = tf.lite.Interpreter(model_content=tflite_model_quant_all)
input_type = interpreter.get_input_details()[0]['dtype']
print(colored('input: ', 'green'), colored(input_type, 'green'))
output_type = interpreter.get_output_details()[0]['dtype']
print(colored('output: ', 'green'), colored(output_type, 'green'))



# ## Saving the models as tflite files

tflite_models_dir = pathlib.Path("./test_models/mnist_tflite_models/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

# Save the unquantized/float model:
tflite_model_file = tflite_models_dir/"mnist_model.tflite"
tflite_model_file.write_bytes(tflite_model)

# Save the quantized models:
tflite_model_quant_file_weights = tflite_models_dir/"mnist_model_quant_weights.tflite"
tflite_model_quant_file_weights.write_bytes(tflite_model_quant_weights)

tflite_model_quant_file_weights_acts = tflite_models_dir/"mnist_model_quant_weights_acts.tflite"
tflite_model_quant_file_weights_acts.write_bytes(tflite_model_quant_weights_acts)

tflite_model_quant_file_all = tflite_models_dir/"mnist_model_quant_all.tflite"
tflite_model_quant_file_all.write_bytes(tflite_model_quant_all)



####################### Part 3 ####################
## Running inferences

# Helper function to run inference on a TFLite model
def run_tflite_model(tflite_file, test_image_indices):
  global test_images

  # Initialize the interpreter
  interpreter = tf.lite.Interpreter(model_path=str(tflite_file), experimental_preserve_all_tensors=True)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  predictions = np.zeros((len(test_image_indices),), dtype=int)
  for i, test_image_index in enumerate(test_image_indices):
    test_image = test_images[test_image_index]
    test_label = test_labels[test_image_index]

    # Check if the input type is quantized, then rescale input data to uint8
    if input_details['dtype'] == np.uint8:
      input_scale, input_zero_point = input_details["quantization"]
      # print(colored('input_scale: ', 'green'), colored(input_scale, 'green'))
      # print(colored('input_zero_point: ', 'green'), colored(input_zero_point, 'green'))
      test_image = test_image / input_scale + input_zero_point

    test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], test_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]

    predictions[i] = output.argmax()

  return predictions


# Change this to test a different image
test_image_index = 0

## Helper function to test the models on one image
def test_model(tflite_file, test_image_index, model_type):
  global test_labels

  predictions = run_tflite_model(tflite_file, [test_image_index])

  plt.imshow(test_images[test_image_index])
  template = model_type + " Model \n True:{true}, Predicted:{predict}"
  _ = plt.title(template.format(true= str(test_labels[test_image_index]), predict=str(predictions[0])))
  plt.grid(True)
  plt.show()
  # img = Image.fromarray(test_images[test_image_index], 'RGB')
  # img.show()

test_model(tflite_model_file, test_image_index, model_type="Float")

test_model(tflite_model_quant_file_weights, test_image_index, model_type="Quantized_Weights")

test_model(tflite_model_quant_file_weights_acts, test_image_index, model_type="Quantized_Weights_Acts")

test_model(tflite_model_quant_file_all, test_image_index, model_type="Quantized_All")

# Helper function to evaluate a TFLite model on all images
def evaluate_model(tflite_file, model_type):
  global test_images
  global test_labels

  test_image_indices = range(test_images.shape[0])
  predictions = run_tflite_model(tflite_file, test_image_indices)

  accuracy = (np.sum(test_labels== predictions) * 100) / len(test_images)

  print('%s model accuracy is %.4f%% (Number of test samples=%d)' % (
      model_type, accuracy, len(test_images)))


evaluate_model(tflite_model_file, model_type="Float")


evaluate_model(tflite_model_quant_file_weights, model_type="Quantized_Weights")

evaluate_model(tflite_model_quant_file_weights_acts, model_type="Quantized_Weights_Acts")

evaluate_model(tflite_model_quant_file_all, model_type="Quantized_All")



####################### Part 4 ####################


# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path = "./test_models/mnist_tflite_models/mnist_model_quant_all.tflite")
interpreter.allocate_tensors()

weight_list = []
for i in range(4,10):
    weight_list.append(interpreter.tensor(i)().transpose())

# weights = np.asarray(weight_list, dtype=object)
# np.save('quantized_weights.npy', weight_list, allow_pickle=True)
