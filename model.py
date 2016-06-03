import tensorflow as tf


class Network:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.state = tf.placeholder(tf.float32, shape=([None] + self.input_shape))

        self.W_conv1 = self._weight([2, 2, self.input_shape[2], 16])
        self.b_conv1 = self._bias([16])
        self.h_conv1 = tf.nn.relu(self._conv(self.state, self.W_conv1, 1) + self.b_conv1)

        self.W_conv2 = self._weight([2, 2, 16, 32])
        self.b_conv2 = self._bias([32])
        self.h_conv2 = tf.nn.relu(self._conv(self.h_conv1, self.W_conv2, 1) + self.b_conv2)

        dim = 1
        for d in self.h_conv2.get_shape()[1:].as_list():
            dim *= d

        self.flat = tf.reshape(self.h_conv2, [-1, dim])

        self.W_flat = self._weight([dim, 128])
        self.b_flat = self._bias([128])
        self.h_flat = tf.nn.relu(tf.matmul(self.flat, self.W_flat) + self.b_flat)

        self.W_output = self._weight([128] + self.output_shape)
        self.b_output = self._bias(self.output_shape)

        self.output = tf.matmul(self.h_flat, self.W_output) + self.b_output

    @staticmethod
    def _weight(shape, stddev=0.01):
        return tf.Variable(tf.truncated_normal(shape, stddev=stddev))

    @staticmethod
    def _bias(shape, value=0.01):
        return tf.Variable(tf.constant(value, shape=shape))

    @staticmethod
    def _conv(input, W, stride):
        return tf.nn.conv2d(input, W, strides=[1, stride, stride, 1], padding='SAME')

    @staticmethod
    def _pool(input):
        return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
