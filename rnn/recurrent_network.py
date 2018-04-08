"""Recurrent Neural Network
Example.
"""
from neologger import Logger
import tensorflow as tf
from tensorflow.contrib import rnn


# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

PROJ_NAME = "RNN"

# Init logger
logger = Logger(PROJ_NAME)

def train():
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
        "out": tf.Variable(tf.random_normal([num_classes]))
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

    # Summary
    log_path = "/tmp/tf"
    summary_step = 10
    tf.summary.scalar("loss", loss_op)
    tf.summary.scalar("Acc", accuracy)
    merged_summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        # Run initializer
        sess.run(init)

        # Writer
        summary_writer = tf.summary.FileWriter(log_path + f"/{PROJ_NAME}", graph=tf.get_default_graph())
        for step in range(1, training_steps + 1):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            # Reshape data to get 28 seq of 28 elements
            batch_x = batch_x.reshape((batch_size, timesteps, num_input))
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % summary_step == 0 or step == 1:
                loss, acc, summary = sess.run([loss_op, accuracy, merged_summary_op], feed_dict={X: batch_x, Y: batch_y})
                summary_writer.add_summary(summary, step)

                if step % display_step == 0 or step ==1:
                    logger.info(f"Step {step}, Minibatch Loss = {loss}, Training Accuracy = {acc}")

        logger.info("Optimization Finished!")
        test_len = 128
        test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
        test_label = mnist.test.labels[:test_len]
        logger.info(f"Test accuracy: {sess.run(accuracy, feed_dict={X: test_data, Y: test_label})}")

if __name__ == "__main__":
    train()
