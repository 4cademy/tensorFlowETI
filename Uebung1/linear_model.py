import tensorflow as tf
import matplotlib . pyplot as plt
class Model(object):
    def __init__(self):
# Initialize the weights to ‘5.0 ‘ and the bias to ‘0.0 ‘
# In practice , these should be initialized to random values ( for example , with ‘tf.random . normal ‘)
        self.W = tf.Variable (5.0)
        self.b = tf.Variable (0.0)
    def __call__(self , x):
        return self .W * x + self .b
model = Model()
assert model(3.0).numpy() == 15.0

def loss (predicted_y , target_y):
    return tf.reduce_mean(tf.square (predicted_y - target_y))

TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000
inputs = tf.random.normal(shape =[ NUM_EXAMPLES ])
noise = tf. random.normal(shape =[ NUM_EXAMPLES ])
outputs = inputs * TRUE_W + TRUE_b + noise

plt.scatter(inputs , outputs , c='b', label = 'Training Data')
plt.scatter(inputs , model ( inputs ), c='r' , label = 'Model Predictions')
plt.legend(loc='upper left')
plt.ylabel('Outputs')
plt.xlabel('Inputs')
plt.show()

print('Current loss: %1.6f' % loss( model(inputs), outputs).numpy())


def train (model , inputs , outputs , learning_rate):
    with tf. GradientTape() as t:
        current_loss = loss (model (inputs), outputs)
    dW , db = t.gradient (current_loss , [model.W, model.b])
    model.W.assign_sub (learning_rate * dW)
    model.b.assign_sub (learning_rate * db)

# Repeatedly run through the training data
model = Model()
# Collect the history of W-values and b-values to plot later
Ws , bs = [], []
epochs = range (10)

for epoch in epochs :
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())
    current_loss = loss(model(inputs), outputs)

    train(model, inputs , outputs , learning_rate = 0.1)
    print ('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' % (epoch , Ws[-1] , bs[-1] , current_loss ))

 # Let ’s plot it all
plt.plot(epochs , Ws , 'r', epochs , bs , 'b')
plt.plot([ TRUE_W ] * len (epochs), 'r--', [ TRUE_b ] * len (epochs), 'b--')
plt.legend ([ 'W', 'b', 'True W', 'True b'])
plt.xlabel ('Epochs')
plt.ylabel ('Value')
plt.show ()


# Plot again the outputs of the trained model
plt.scatter(inputs , outputs , c='b')
plt.scatter(inputs , model (inputs), c='r')
plt.legend(['Training Data', 'Prediction'])
plt.xlabel('Inputs')
plt.ylabel('Outputs')
plt.show()
print('Current loss : %1.6f' % loss (model (inputs), outputs).numpy ())
