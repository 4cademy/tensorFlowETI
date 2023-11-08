import tensorflow as tf
from datetime import datetime
from packaging import version
from tensorflow import keras
import tensorboard

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
"This notebook requires TensorFlow 2.0 or above."

# rm -rf ./logs/

@tf.function
def my_func(x, y):
	return tf.add(x, y)

stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = 'logs/func/%s' % stamp
writer = tf.summary.create_file_writer(logdir)

tf.summary.trace_on(graph=True, profiler=True)
x = 2
y = 4
z = my_func(x, y)
print(z) #tf.Tensor(6, shape=(), dtype=int32)

with writer.as_default():
	tf.summary.trace_export(
		name="my_func_trace",
		step=0,
		profiler_outdir=logdir)
