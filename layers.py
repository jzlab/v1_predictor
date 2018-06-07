import tensorflow as tf
import numpy as np
import math

class Linear(object):
    def __init__(self, ncell, input_shape):
        numin = input_shape[-1]
        input_var = tf.placeholder(tf.float32, shape=input_shape)

        ## linear layer
        with tf.variable_scope('linear'):
            print("building linear layer")

            weights = tf.get_variable('weights',
                initializer = tf.truncated_normal([numin, ncell],
                stddev=0.1 / math.sqrt(float(numin))))

            biases = tf.get_variable('biases', initializer = tf.zeros([ncell]))

            #hidden2_drop = tf.nn.dropout(hidden2, keep_prob)
            #linear = tf.nn.relu(tf.add(tf.matmul(hidden2_drop, weights),biases))

            linear = tf.add(tf.matmul(input_var, weights), biases)
            print("linear mapping %d elemements to %d elements" % (numin, ncell))

        print(tf.shape(linear))
        self.output = linear

        self.output_shape = linear.get_shape()[1].value
