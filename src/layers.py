import tensorflow as tf
import numpy as np
import math

from keras.layers import Input

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

class Sigmoid():

    def __init__(self,input_var,ncell=None):
        ## add parametrized sigmoid nonlinearity to output

        if ncell is None:
            ncell = input_var.get_shape()[1].value

        with tf.variable_scope('nonlinear'):

            print("building non linear layer")
            weights = tf.get_variable('weights',
                initializer = .1+tf.truncated_normal([ncell],stddev=0.02))

            biases = tf.get_variable('biases', initializer = tf.zeros([ncell]))

            #hidden2_drop = tf.nn.dropout(hidden2, keep_prob)
            #linear = tf.nn.relu(tf.add(tf.matmul(hidden2_drop, weights),biases))

            diagweights = tf.diag(weights)

            print("linear: ", str(input_var))
            print("weights: ", str(weights))
            print("diagweights: ", str(diagweights))

            #siglin = tf.sigmoid(linear)
            #nonlindum = tf.sigmoid(linear)

            nonlindum = tf.nn.relu(input_var)
            sigmatmul = tf.matmul(nonlindum, diagweights)
            nonlinear = tf.add(sigmatmul ,biases)
            #print(" adding nonlinearity the %d outputs" % (ncell))

        self.output = nonlinear
