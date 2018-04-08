"""Recurrent Neural Network
Example.
"""
from neologger import Logger
import tensorflow as tf
from tensorflow.contrib import rnn


# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

# Init logger
logger = Logger(__name__)
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# Training Parameters
learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 200

# Network Parameters
num_input = 28
timesteps = 28
num_hidden = 128
num_classes = 10

# tf graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Define weight
weight = {
    "out": tf.Variable(tf.random_normal([num_hidden, num_classes]))
}

biases = {
    "out": tf.Variable(tf.random_normal([num_classes])
}


def RNN(x, weights, biases):

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
    x = tf.unstack(x, timesteps, 1)

    # Define lstm cell
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, sates = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = RNN(X, weight, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model (with test logitsm for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Init the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


with tf.Session() as sess:

    # Run initializer
    sess.run(init)

    for step in range(1, traing_step + 1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)

        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step ==1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
            logger.info(f"Stet {step}, Minibatch Loss = {loss}, Training Accuracy = {acc}")
    logger.info("Optimization Finished!")

