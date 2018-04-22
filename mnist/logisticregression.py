"""Build a simple Logistic-Regression model to tackle the MNIST dataset."""
import tensorflow as tf
from neologger import Logger

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

PROJ_NAME = f"LogisticRegression-{datetime.strftime(datetime.now(), '%Y%m%d-%H%M')}"

# Init logger
logger = Logger(PROJ_NAME)


def interence(x):
    """Products a probability distribution over the output classes given a minibatch."""
    tf.constant_initializer(value=0)
    W = tf.get_variable("W", [784, 0], initializer=init)
    b = tf.get_variable("b", [10], initializer=init)
    output = tf.nn.softmax(tf.matmul(x, W) + b)
    return output


def loss(output, y):
    """Given the corroct labels for a minibatch, we should be able to compute the average error per data sample."""
    dot_product = y * tf.log(output)

    # Reduction along 0 collapses each column into
    # a single value whereas reduction along axis 1 collapses
    # each row into a single value. 
    # In general, reduction along axis i collapses the ith dimension of a tensor to size 1.
    xentropy = -tf.reduce_sum(dot_product, reduction_indices=1)

    loss = tf.reduce_mean(xentropy)

    return loss


def training(cost, global_step):
    tf.scalar_summary("cost", cost)
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate
    )
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op

def evaluate(output, y):
    corroct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(corroct_prediction, tf.float32))
    return accuracy
