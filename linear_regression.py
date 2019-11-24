import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

top_scope = tf.get_variable_scope()


def load_data():
    from sklearn.datasets import load_boston
    boston_dataset = load_boston()
    X = np.array(boston_dataset.data)
    y = np.array(boston_dataset.target)
    return X, y


def model(x: tf.Tensor):
    """
    linear regression model: y_predict = W*x + b
    please use your matrix multiplication implementation.
    :param x: symbolic tensor with shape (batch, dim)
    :return:  a tuple that contains: 1.symbolic tensor y_predict, 2. list of the variables used in the model: [W, b]
                the result shape is (batch)
    """
    with tf.variable_scope(top_scope, reuse=tf.AUTO_REUSE, dtype=tf.float64):
        w = tf.get_variable(name='weights', initializer=tf.random_normal(shape=[int(x.shape[1]), 1], dtype=tf.float64))
        b = tf.get_variable(name='bias', initializer=tf.random_normal(shape=[1], dtype=tf.float64))
        return tf.reshape(tf.matmul(x, w), shape=[x.shape[0]]) + b, [w, b]


def train(epochs, learning_rate, batch_size):
    """
    create linear regression using model() function from above and train it on boston houses dataset using batch-SGD.
    please normalize your data as a pre-processing step.
    please use your mse-loss implementation.
    :param epochs: number of epochs
    :param learning_rate: the learning rate of the SGD
    :return: list contains the mean loss from each epoch.
    """
    X, y = load_data()
    normalize(X, copy=False)  # normalize data
    normalize(np.reshape(y, newshape=(-1, 1)), copy=False)
    num_samples = X.shape[0]
    num_of_batches = int(num_samples / batch_size)

    # declare placeholders
    X_ph = tf.placeholder(dtype=tf.float64, shape=[batch_size, X.shape[1]])
    Y_ph = tf.placeholder(dtype=tf.float64, shape=[batch_size])

    # get model, loss and gradients
    y_pred, [w, b] = model(tf.convert_to_tensor(X_ph))
    loss = tf.losses.mean_squared_error(Y_ph, y_pred)
    gradients = tf.gradients(loss, [w, b])
    # update operations
    updated_w_op = w.assign(w - gradients[0] * learning_rate)
    updated_b_op = b.assign(b - gradients[1] * learning_rate)

    losses = []
    with tf.Session() as sess:
        # init variables
        init = tf.global_variables_initializer()
        sess.run(init)
        for ep in range(epochs):
            epoch_loss = 0.0
            for batch in range(num_of_batches):
                # get random batch
                indices = np.random.randint(num_samples, size=batch_size)
                rand_x = X[indices]
                rand_y = y[indices]

                y_pred, [w, b] = model(tf.convert_to_tensor(rand_x))
                _, batch_loss, _ = sess.run([y_pred, loss, gradients], feed_dict={X_ph: rand_x, Y_ph: rand_y})
                sess.run([updated_w_op, updated_b_op], feed_dict={X_ph: rand_x, Y_ph: rand_y})
                # save the loss of the current batch
                epoch_loss += batch_loss / num_of_batches
            # save loss
            losses.append(epoch_loss)
            # print("epoch=" + str(ep) + ", loss=" + str(epoch_loss))

    return losses


def main():
    losses = train(50, 0.01, 32)
    plt.plot(losses)
    plt.show()
    pass


if __name__ == "__main__":
    main()
