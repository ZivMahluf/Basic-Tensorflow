import tensorflow as tf
import numpy as np

# Network Parameters
n_hidden = 100  # 1st layer number of neurons
n_input = 784  # MNIST data input (img shape: 28*28)
top_scope = tf.get_variable_scope()
keep_prob = 0.8


def mlp(x: tf.Tensor, nlabels):
    """
    multi layer perceptrone: x -> linear > relu > linear.
    :param x: symbolic tensor with shape (batch, 28, 28)
    :param nlabels: the dimension of the output.
    :return: a symbolic tensor, the result of the mlp, with shape (batch, nlabels).
    the model return logits (before softmax).
    """
    with tf.variable_scope(top_scope, reuse=tf.AUTO_REUSE):
        flatx = tf.reshape(x, shape=[tf.shape(x)[0], n_input])
        # first linear layer
        w = tf.get_variable(name='mlp_weights', initializer=tf.random_normal(shape=[n_input, n_hidden]), trainable=True)
        b = tf.get_variable(name='mlp_bias', initializer=tf.random_normal(shape=[n_hidden]), trainable=True)
        lin_layer = tf.add(tf.matmul(flatx, w), b)
        # relu layer
        relu_layer = tf.nn.relu(lin_layer)
        # second linear layer
        w2 = tf.get_variable(name='mlp_weights2', initializer=tf.random_normal(shape=[n_hidden, nlabels]),
                             trainable=True)
        b2 = tf.get_variable(name='mlp_bias2', initializer=tf.random_normal(shape=[nlabels]), trainable=True)
        lin_layer2 = tf.add(tf.matmul(relu_layer, w2), b2)
        # regularization = tf.nn.dropout(lin_layer2, keep_prob)
        return lin_layer2


def conv_net(x: tf.Tensor, nlabels):
    """
    convnet.
    in the convolution use 3x3 filteres with 1x1 strides, 20 filters each time.
    in the  maxpool use 2x2 pooling.
    :param x: symbolic tensor with shape (batch, 28, 28)
    :param nlabels: the dimension of the output.
    :return: a symbolic tensor, the result of the mlp, with shape (batch, nlabels).
    the model return logits (before softmax).
    """
    with tf.variable_scope(top_scope, reuse=tf.AUTO_REUSE):
        filters = {
            'filter1': tf.get_variable(name='filter1', initializer=tf.random_normal(shape=[3, 3, 1, 20]),
                                       trainable=True),
            'filter2': tf.get_variable(name='filter2', initializer=tf.random_normal(shape=[3, 3, 20, 20]),
                                       trainable=True)
        }
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        conv1 = tf.nn.conv2d(x, filter=filters['filter1'], strides=[1, 1, 1, 1], padding='SAME')
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
        conv2 = tf.nn.conv2d(pool1, filter=filters['filter2'], strides=[1, 1, 1, 1], padding='SAME')
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
        before_linear = tf.reshape(pool2, [-1, 28 * 28 * 20])
        # linear layer
        w = tf.get_variable(name='conv_weights', initializer=tf.random_normal(shape=[28 * 28 * 20, nlabels]))
        b = tf.get_variable(name='conv_bias', initializer=tf.random_normal(shape=[nlabels]))
        out_layer = tf.add(tf.matmul(before_linear, w), b)
        return out_layer
