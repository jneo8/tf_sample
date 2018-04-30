"""Building a ReLU with two hidden layer Model for MNIST in TensorFlow."""
from datetime import datetime
import tensorflow as tf
from neologger import Logger

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

PROJ_NAME = f"MNIST-CNN"
LOG_PATH = "/tmp/tf" + f"/{PROJ_NAME}"

# Init logger
logger = Logger(PROJ_NAME)


def main():
    """Main."""
    # Data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # Paramters
    learning_rate = 0.01
    training_epochs = 500
    batch_size = 100
    display_step = 1

    def conv2d(input, weight_shape, bias_shape):
        """Conv2d layer."""
        incoming = weight_shape[0] * weight_shape[1] * weight_shape[2]
        weight_init = tf.random_normal_initializer(stddev=(2.0 / incoming) ** 0.5)
        W = tf.get_variable("W", weight_shape, initializer=weight_init)
        bias_init = tf.constant_initializer(value=0)
        b = tf.get_variable("b", bias_shape, initializer=bias_init)
        conv_out = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding="SAME")

        return tf.nn.relu(tf.nn.bias_add(conv_out, b))

    def max_pool(input, k=2):
        return tf.nn.max_pool(
            input,
            ksize=[1, k, k, 1],
            strides=[1, k, k, 1],
            padding="SAME",
        )

    def layer(input, weight_shape, bias_shape):
        """Layer."""
        weight_stddev = (2.0 / weight_shape[0])**0.5
        w_init = tf.random_normal_initializer(stddev=weight_stddev)
        bias_init = tf.constant_initializer(value=0)
        W = tf.get_variable("W", weight_shape, initializer=w_init)
        b = tf.get_variable("b", bias_shape, initializer=bias_init)
        return tf.nn.relu(tf.matmul(input, W) + b)

    def evaluate(output, y):
        """Evaluate accuracy."""
        corroct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(corroct_prediction, tf.float32))
        tf.summary.scalar("Acc", accuracy)
        return accuracy


    def inference(x, keep_prob):
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        with tf.variable_scope("conv_1"):
            conv_1 = conv2d(x, [5, 5, 1, 32], [32])
            pool_1 = max_pool(conv_1)

        with tf.variable_scope("conv_2"):
            conv_2 = conv2d(pool_1, [5, 5, 32, 64], [64])
            pool_2 = max_pool(conv_2)

        with tf.variable_scope("fc"):
            pool_2_flat = tf.reshape(pool_2, [-1, 7 * 7 * 64])
            fc_1 = layer(pool_2_flat, [7 * 7 * 64, 1024], [1024])
            # apply dropout
            fc_1_drop = tf.nn.dropout(fc_1, keep_prob)

        with tf.variable_scope("output"):
            output = layer(fc_1_drop, [1024, 10], [10])

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

    with tf.Graph().as_default():

        with tf.variable_scope("conv_model"):
            # mnist data image of shape 28*28=784
            x = tf.placeholder("float", [None, 784])

            # 0-9 digits recognition -> 10 classes
            y = tf.placeholder("float", [None, 10])

            keep_prob = tf.placeholder(tf.float32)

            # Define network
            output = inference(x, keep_prob=keep_prob)
            cost = loss(output, y)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            train_op = training(cost, global_step)

            # define evaluate optimizer & summary optimizer
            eval_op = evaluate(output, y)
            summary_op = tf.summary.merge_all()

            # Init
            sess = tf.Session()
            summary_writer = tf.summary.FileWriter(LOG_PATH, graph=sess.graph_def)
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            tf.train.start_queue_runners(sess=sess)
            for epoch in range(training_epochs):

                avg_cost = 0.
                total_batch = int(mnist.train.num_examples / batch_size)

                # Loop over all batches
                for i in range(total_batch):
                    mbatch_x, mbatch_y = mnist.train.next_batch(batch_size)
                    # Fit training using batch data
                    feed_dict = {x: mbatch_x, y: mbatch_y, keep_prob: 0.5}
                    sess.run(train_op, feed_dict=feed_dict)
                    # Compute average loss
                    minibatch_cost = sess.run(cost, feed_dict=feed_dict)
                    avg_cost += minibatch_cost / total_batch

                # Display logs pre epoch step
                if epoch % display_step == 0:
                    accuracy = sess.run(
                        eval_op,
                        feed_dict={
                            x: mnist.validation.images,
                            y: mnist.validation.labels,
                            keep_prob: 1,
                        }
                    )
                    logger.debug(f"Epoch: {epoch} Validation Error: {(1 - accuracy)}")
                    summary_str = sess.run(
                        summary_op,
                        feed_dict={
                            x: mnist.validation.images,
                            y: mnist.validation.labels,
                            keep_prob: 0.5,
                        }
                    )
                    summary_writer.add_summary(summary_str, sess.run(global_step))

            logger.info("Optimization Finished!!")

            test_feed_dict = {
                x: mnist.test.images,
                y: mnist.test.labels,
                keep_prob: 0.5,
            }

            accuracy = sess.run(eval_op, feed_dict=test_feed_dict)
            logger.info(f"Test Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
