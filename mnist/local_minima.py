"""Building a ReLU with two hidden layer Model for MNIST in TensorFlow."""
from datetime import datetime

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
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
        """Define output."""
        with tf.variable_scope("hidden_1"):
            hidden_1 = layer(x, [784, 256], [256])
        with tf.variable_scope("hidden_2"):
            hidden_2 = layer(hidden_1, [256, 256], [256])
        with tf.variable_scope("output"):
            output = layer(hidden_2, [256, 10], [10])
        return output

    def loss(output, y):
        """Given the corroct labels for a minibatch, we should be able to compute the average error per data sample."""

        # Find x_entropy
        x_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)

        # find loss
        loss = tf.reduce_mean(x_entropy)

        tf.summary.scalar("loss", loss)
        return loss

    def training(cost, global_step):
        """Return training optimizer."""
        tf.summary.scalar("cost", cost)
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate
        )
        train_op = optimizer.minimize(cost, global_step=global_step)
        return train_op

    def evaluate(output, y):
        """Evaluate accuracy."""
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

        # define evaluate optimizer & summary optimizer
        eval_op = evaluate(output, y)
        summary_op = tf.summary.merge_all()

        # Init
        sess = tf.Session()
        summary_writer = tf.summary.FileWriter(LOG_PATH, graph=sess.graph_def)
        init_op = tf.initialize_all_variables()
        sess.run(init_op)

        with tf.variable_scope("mlp_model") as scope:
            output_opt = inference(x)
            cost_opt = loss(output_opt, y)
            scope.reuse_variables()
            var_list_opt = [
                "hidden_1/W",
                "hidden_1/b",
                "hidden_2/W",
                "hidden_2/b",
                "output/W",
                "output/b",
            ]
            var_list_opt = [tf.get_variable(v) for v in var_list_opt]

        with tf.variable_scope("mlp_init") as scope:
            output_rand = inference(x)
            cost_rand = loss(output_rand, y)
            scope.reuse_variables()
            var_list_rand = [
                "hidden_1/W",
                "hidden_1/b",
                "hidden_2/W",
                "hidden_2/b",
                "output/W",
                "output/b",
            ]
            var_list_rand = [tf.get_variable(v) for v in var_list_rand]
            init_op = tf.initialize_variables(var_list_rand)
            sess.run(init_op)

        with tf.variable_scope("mlp_inter") as scope:
            alpha = tf.placeholder("float", [1, 1])
            beta = 1 - alpha
            h1_W_inter = var_list_opt[0] * beta + var_list_rand[0] * alpha
            h1_b_inter = var_list_opt[1] * beta + var_list_rand[1] * alpha
            h2_W_inter = var_list_opt[2] * beta + var_list_rand[2] * alpha
            h2_b_inter = var_list_opt[3] * beta + var_list_rand[3] * alpha
            o_W_inter = var_list_opt[4] * beta + var_list_rand[4] * alpha
            o_b_inter = var_list_opt[5] * beta + var_list_rand[5] * alpha

            h1_inter = tf.nn.relu(tf.matmul(x, h1_W_inter) + h1_b_inter)
            h2_inter = tf.nn.relu(tf.matmul(h1_inter, h2_W_inter) + h2_b_inter)
            o_inter = tf.nn.relu(tf.matmul(h2_inter, o_W_inter) + o_b_inter)

            cost_inter = loss(o_inter, y)

                # For loop in each epochs.
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

        results = []
        for a in np.range(-2, 2, 0.01):
            feed_dict = {
                x: mnist.test.images,
                y: mnist.test.labels,
                alpha: [[a]],
            }
            cost, summary_str = sess.run([cost_inter, summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, (a + 2) / 0.01)
            results.append(cost)
        plt.plot(np.arrange(-2, 2, 0.01), result, 'ro')
        plt.ylabel("Incurred Error")
        plt.xlabel("Alpha")
        plt.savefig(f"{LOG_PATH}/{alpha_img.png}")



if __name__ == "__main__":
    main()
