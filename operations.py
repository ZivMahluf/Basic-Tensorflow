import tensorflow as tf
import numpy as np


# ----------------- relu example --------------------

# the relu itself
def relu_numpy(x: np.ndarray):
    result = np.zeros_like(x)
    result[x > 0] = x[x > 0]
    return result


# the relu gradient
def relu_grad_numpy(x: np.ndarray, dy: np.ndarray):
    # x and y should have the same shapes.
    result = np.zeros_like(x)
    result[x > 0] = dy[x > 0]
    return result


# the relu tensorflow operation
@tf.custom_gradient
def relu_tf(x):
    result = tf.py_func(relu_numpy, [x], tf.float64, name='my_relu_op')

    def grad(dy):
        return tf.py_func(relu_grad_numpy, [x, dy], tf.float64, name='my_relu_grad_op')

    return result, grad


# ----------------- batch matrix multiplication --------------
# a.shape = (n, k)
# b.shape = (k, m)
def matmul_numpy(a: np.ndarray, b: np.ndarray):
    result = a @ b  # YOUR CODE HERE

    return result


# dy_dab.shape = (a.shape[0], b.shape[1])
# dy_da.shape = a.shape
# dy_db.shape = b.shape
def matmul_grad_numpy(a: np.ndarray, b: np.ndarray, dy_dab: np.ndarray):
    dy_da = dy_dab @ b.T
    dy_db = a.T @ dy_dab
    return [dy_da, dy_db]


@tf.custom_gradient
def matmul_tf(a, b):
    # use tf.numpy_function

    result = tf.py_func(matmul_numpy, [a, b], tf.float64, name='my_matmul_tf_op')  # YOUR CODE HERE

    def grad(dy_dab):
        return tf.py_func(matmul_grad_numpy, [a, b, dy_dab], tf.float64, name='my_matmul_grad_op')  # YOUR CODE HERE

    return result, grad


# ----------------- mse loss --------------
# y.shape = (batch)
# ypredict.shape = (batch)
# the result is a scalar
# dloss_dyPredict.shape = (batch)
def mse_numpy(y, ypredict):
    loss = np.mean(np.power(y - ypredict, 2))
    return loss


def mse_grad_numpy(y, yPredict,
                   dy):  # dy is gradient from next node in the graph, not the gradient of our y!
    dloss_dyPredict = np.mean(-2 * (y - yPredict))
    dloss_dy = np.mean(2 * (y - yPredict))
    return [dloss_dy, dloss_dyPredict]


@tf.custom_gradient
def mse_tf(y, y_predict):
    # use tf.numpy_function

    loss = tf.py_func(mse_numpy, [y, y_predict], tf.float64, name='my_mse_tf_op')

    def grad(dy):
        return tf.py_func(mse_grad_numpy, [y, y_predict, dy], tf.float64, name='my_mse_grad_op')

    return loss, grad
