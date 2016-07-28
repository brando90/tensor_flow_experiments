# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

pickle_file = '/home/maxkhk/Documents/Udacity/DeepLearningCourse/SourceCode/tensorflow/examples/udacity/notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


#for NeuralNetwork model code is below
#We will use SGD for training to save our time. Code is from Assignment 2
#beta is the new parameter - controls level of regularization.
#Feel free to play with it - the best one I found is 0.001
#notice, we introduce L2 for both biases and weights of all layers

batch_size = 128
beta = 0.001

#building tensorflow graph
graph = tf.Graph()
with graph.as_default():
      # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  #introduce batchnorm
  tf_train_dataset_bn = tf.contrib.layers.batch_norm(tf_train_dataset)


  #now let's build our new hidden layer
  #that's how many hidden neurons we want
  num_hidden_neurons = 1024
  #its weights
  hidden_weights = tf.Variable(
    tf.truncated_normal([image_size * image_size, num_hidden_neurons]))
  hidden_biases = tf.Variable(tf.zeros([num_hidden_neurons]))

  #now the layer itself. It multiplies data by weights, adds biases
  #and takes ReLU over result
  hidden_layer = tf.nn.relu(tf.matmul(tf_train_dataset_bn, hidden_weights) + hidden_biases)

  #adding the batch normalization layerhi()
  hidden_layer_bn = tf.contrib.layers.batch_norm(hidden_layer)

  #time to go for output linear layer
  #out weights connect hidden neurons to output labels
  #biases are added to output labels
  out_weights = tf.Variable(
    tf.truncated_normal([num_hidden_neurons, num_labels]))

  out_biases = tf.Variable(tf.zeros([num_labels]))

  #compute output
  out_layer = tf.matmul(hidden_layer_bn,out_weights) + out_biases
  #our real output is a softmax of prior result
  #and we also compute its cross-entropy to get our loss
  #Notice - we introduce our L2 here
  loss = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    out_layer, tf_train_labels) +
    beta*tf.nn.l2_loss(hidden_weights) +
    beta*tf.nn.l2_loss(hidden_biases) +
    beta*tf.nn.l2_loss(out_weights) +
    beta*tf.nn.l2_loss(out_biases)))

  #now we just minimize this loss to actually train the network
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

  #nice, now let's calculate the predictions on each dataset for evaluating the
  #performance so far
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(out_layer)
  valid_relu = tf.nn.relu(  tf.matmul(tf_valid_dataset, hidden_weights) + hidden_biases)
  valid_prediction = tf.nn.softmax( tf.matmul(valid_relu, out_weights) + out_biases)

  test_relu = tf.nn.relu( tf.matmul( tf_test_dataset, hidden_weights) + hidden_biases)
  test_prediction = tf.nn.softmax(tf.matmul(test_relu, out_weights) + out_biases)



#now is the actual training on the ANN we built
#we will run it for some number of steps and evaluate the progress after
#every 500 steps

#number of steps we will train our ANN
num_steps = 3001

#actual training
with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
      print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
