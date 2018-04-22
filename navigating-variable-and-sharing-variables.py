import tensorflow as tf
from neologger import Logger

logger = Logger(__name__)

def my_network_error(input):
    W_1 = tf.Variable(tf.random_uniform([784, 100], -1, 1), name="W_1")
    b_1 = tf.Variable(tf.zeros([100]), name="biases_1")
    output_1 = tf.matmul(input, W_1) + b_1


    W_2 = tf.Variable(tf.random_uniform([100, 50], -1, 1), name="W_2")
    b_2 = tf.Variable(tf.zeros([50]), name="biases_2")
    output_2 = tf.matmul(output_1, W_2) + b_2

    W_3 = tf.Variable(tf.random_uniform([50, 10], -1, 1), name="W_3")
    b_3 = tf.Variable(tf.zeros([10]), name="biases_3")
    output_3 = tf.matmul(output_2, W_3) + b_3

    logger.debug(f"{W_1.name} {W_2.name} {W_3.name}")
    logger.debug(f"{b_1.name} {b_2.name} {b_3.name}")

def main_error():
    # When we try to use this network on two different input, we get something unexpected:
    # If We observe closely, our second call to my_network doesn't use the same variables as the first call.
    # In fact, the namees are different.
    # Instead we're created a second set of variables!
    i_1 = tf.placeholder(tf.float32, [1000, 784], name="i_1")
    my_network_error(i_1)
    i_2 = tf.placeholder(tf.float32, [1000, 784], name="i_2")
    my_network_error(i_2)



def layer(input, weight_shape, bias_shape):
    weight_init = tf.random_uniform_initializer(minval=-1, maxval=1)
    bias_init = tf.constant_initializer(value=0)

    # Unlike tf.Variable, the tf.get_variable command checks that a variable of the given name has't already been instantiated.
    W = tf.get_variable("W", weight_shape, initializer=weight_init)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    logger.debug(f"{W.name} {b.name}")
    return tf.matmul(input, W) + b

def my_network(input):
    with tf.variable_scope("layer_1"):
        output_1 = layer(input, [784, 100], [100])
    with tf.variable_scope("layer_2"):
        output_2= layer(output_1, [100, 50], [50])
    with tf.variable_scope("layer_3"):
        output_3= layer(output_2, [50, 10], [10])

    return output_3

def main():
    # By default, sharing is not allowed, but if we want to enable sharing within variable scope, we can say so explicitly.
    with tf.variable_scope("shared_varibles") as scope:
        i_1 = tf.placeholder(tf.float32, [1000, 784], name="i_1")
        my_network(i_1)
        scope.reuse_variables()
        i_2 = tf.placeholder(tf.float32, [1000, 784], name="i_2")
        my_network(i_2)


if __name__ == "__main__":
    main_error()
    main()
