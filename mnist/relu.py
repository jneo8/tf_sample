"""Building a ReLU with two hidden layer Model for MNIST in TensorFlow."""
from datetime import datetime

import tensorflow as tf
from neologger import Logger

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

PROJ_NAME = f"MNIST-ReLU-with-two-hidden-layer-{datetime.strftime(datetime.now(), '%Y%m%d-%H%M')}"
LOG_PATH = "/tmp/tf" + f"/{PROJ_NAME}"

# Init logger
logger = Logger(PROJ_NAME)


def main():
    """Main."""
    # Data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # Paramters
    learning_rate = 0.01
    training_epochs = 10
    batch_size = 100
    display_step = 1

    def layer(input, weight_shape, bias_shape):
        """Layer."""
        weight_stddev = (2.0 / weight_shape[0])**0.5
        w_init = tf.random_normal_initializer(stddev=weight_stddev)
        bias_init = tf.constant_initializer(value=0)
        W = tf.get_variable("W", weight_shape, initializer=w_init)
        b = tf.get_variable("b", bias_shape, initializer=bias_init)
        return tf.nn.relu(tf.matmul(input, W) + b)

    # Define Logistic Regression
    def inference(x):
        """Products a probability distribution over the output classes given a minibatch."""
        with tf.variable_scope("hidden_1"):
            hidden_1 = layer(x, [784, 256], [256])
        with tf.variable_scope("hidden_2"):
            hidden_2 = layer(hidden_1, [256, 256], [256])
        with tf.variable_scope("output"):
            output = layer(hidden_2, [256, 10], [10])
        return output

    def loss(output, y):
        """Given the corroct labels for a minibatch, we should be able to compute the average error per data sample."""
        xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)
        loss = tf.reduce_mean(xentropy)
        return loss

    def training(cost, global_step):
        tf.summary.scalar("cost", cost)
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate
        )
        train_op = optimizer.minimize(cost, global_step=global_step)
        return train_op

    def evaluate(output, y):
        corroct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(corroct_prediction, tf.float32))
        tf.summary.scalar("Acc", accuracy)
        return accuracy

    with tf.Graph().as_default():
        # mnist data image of shape 28*28=784
        x = tf.placeholder("float", [None, 784])

        # 0-9 digits recognition -> 10 classes
        y = tf.placeholder("float", [None, 10])

        output = inference(x)
        cost = loss(output, y)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = training(cost, global_step)
        eval_op = evaluate(output, y)
        summary_op = tf.summary.merge_all()

        sess = tf.Session()

        summary_writer = tf.summary.FileWriter(LOG_PATH, graph=sess.graph_def)
        init_op = tf.initialize_all_variables()
        sess.run(init_op)

        for epoch in range(training_epochs):

            avg_cost = 0.
            total_batch = int(mnist.train.num_examples / batch_size)

            # Loop over all batches
            for i in range(total_batch):
                mbatch_x, mbatch_y = mnist.train.next_batch(batch_size)
                # Fit training using batch data
                feed_dict = {x: mbatch_x, y: mbatch_y}
                sess.run(train_op, feed_dict=feed_dict)
                # Compute average loss
                minibatch_cost = sess.run(cost, feed_dict=feed_dict)
                avg_cost += minibatch_cost / total_batch

            # Display logs pre epoch step
            if epoch % display_step == 0:
                val_feed_dict = {
                    x: mnist.validation.images,
                    y: mnist.validation.labels,
                }
                accuracy = sess.run(eval_op, feed_dict=val_feed_dict)
                logger.debug(f"Epoch: {epoch} Validation Error: {(1 - accuracy)}")
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, sess.run(global_step))

        logger.info("Optimization Finished!!")

        test_feed_dict = {
            x: mnist.test.images,
            y: mnist.test.labels,
        }

        accuracy = sess.run(eval_op, feed_dict=test_feed_dict)
        logger.info(f"Test Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
