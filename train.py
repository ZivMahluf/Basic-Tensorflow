import tensorflow as tf
from model import mlp, conv_net
import numpy as np
import matplotlib.pyplot as plt
from plot_graphs import plot_graphs

training_epochs = 25
default_learning_rate = 0.001
adversarial_samples = 10  # use only part of the data, for runtime costs
label_dict = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

# placeholders
X = tf.placeholder("float", [None, 28, 28])
Y = tf.placeholder("float", [None, 10])
top_scope = tf.get_variable_scope()


def load_data():
    return tf.keras.datasets.fashion_mnist.load_data()


def display_adversarial_images(adv_images, data_images, labels, random_indices):
    # plt.gray()
    for index in random_indices:
        fig, axarr = plt.subplots(1, 2)
        fig.suptitle("label is: " + str(labels[index]))
        axarr[0].imshow(data_images[index])
        axarr[1].imshow(adv_images[index])
        plt.savefig('adv_images/' + str(index) + '.png', format='png')
        plt.cla()


def create_adversarial_pattern(sess, logits, model_fn, input_image, input_label, num_labels):
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    # loss_object = tf.nn.softmax_cross_entropy_with_logits()

    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model_fn(input_image, num_labels)
        prediction_reshaped = tf.reshape(prediction, shape=[num_labels])
        # loss = tf.nn.softmax_cross_entropy_with_logits(labels=input_label, logits=prediction_reshaped)
        loss = loss_object(input_label, prediction_reshaped)

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_image)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    return signed_grad


def train_adversarial(sess, logits, model_fn, data, labels, num_labels):
    # preprocess data
    data = tf.convert_to_tensor(data, dtype=tf.float32)
    data = data / 255
    labels = tf.convert_to_tensor(labels)
    data = tf.reshape(data, shape=[-1, 1, 28, 28])  # we will run on 1 image at a time
    adv_data = []
    eps = 0.7
    # update all data
    for i in range(data.shape[0]):
        if i % 100 == 0: print(i)
        signed_grad = create_adversarial_pattern(sess, logits, model_fn, data[i], labels[i], num_labels)
        real_signed_grad = sess.run(signed_grad)
        real_data_i = sess.run(data[i])
        adv_data.append(real_data_i + eps * real_signed_grad)

    adv_data = tf.convert_to_tensor(np.array(adv_data), dtype=tf.float32)
    adv_data = tf.reshape(adv_data, shape=[-1, 28, 28])
    data = tf.reshape(data, shape=[-1, 28, 28])
    print("finished adv data")
    # check accuracy
    adv_pred = tf.nn.softmax(model_fn(adv_data, num_labels))  # Apply softmax to adversarial prediction
    correct_prediction = tf.equal(tf.argmax(adv_pred, 1), tf.argmax(labels, 1))
    # Calculate accuracy
    adv_accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    adv_acc = sess.run(adv_accuracy)
    print("adversarial accuracy:" + str(adv_acc))

    random_indices = range(adversarial_samples)
    adv_images, data_images, labels_images = {}, {}, {}
    for index in random_indices:
        adv_images[index], data_images[index], numbered_label = sess.run(
            [adv_data[index], data[index], labels[index]])
        labels_images[index] = label_dict[int(np.argmax(numbered_label, axis=0))]

    display_adversarial_images(adv_images, data_images, labels_images, random_indices)


def train(model_fn, batch_size, learning_rate=None):
    """
    load FashionMNIST data.
    create model using model_fn, and train it on FashionMNIST.
    :param model_fn: a function to create the model (should be one of the functions from model.py)
    :param batch_size: the batch size for the training
    :param learning_rate: optional parameter - option to specify learning rate for the optimizer.
    :return:
    """
    (x_train, y_train), (x_test, y_test) = load_data()

    # preprocess data - convert y_train and y_test to One Hot, and convert X's values to float32
    max_y_value = 10
    new_y_train = np.zeros((y_train.size, max_y_value))
    new_y_train[np.arange(y_train.size), y_train] = 1
    new_y_test = np.zeros((y_test.size, max_y_value))
    new_y_test[np.arange(y_test.size), y_test] = 1

    x_train.astype(dtype=np.float32, copy=False)
    x_test.astype(dtype=np.float32, copy=False)

    # get model
    logits = model_fn(X, 10)
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    if learning_rate:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate=default_learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Initializing the variables
    init = tf.global_variables_initializer()
    num_samples = int(x_train.shape[0])
    train_losses, train_accuracy, test_losses, test_accuracy = [], [], [], []

    # Get model accuracy estimator
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # train model
    with tf.Session() as sess:
        sess.run(init)
        total_batch = int(int(num_samples) / batch_size)
        for ep in range(training_epochs):
            avg_cost = 0.
            # Loop over all batches
            for batch in range(total_batch):
                indices = np.random.randint(num_samples, size=batch_size)
                rand_x = x_train[indices]
                rand_y = new_y_train[indices]

                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([train_op, loss_op], feed_dict={X: rand_x, Y: rand_y})
                # Compute average loss
                avg_cost += c / total_batch
            # save losses and accuracies for the current epoch
            train_losses.append(avg_cost)
            train_accuracy.append(accuracy.eval({X: x_train, Y: new_y_train}))
            test_loss = sess.run(loss_op, feed_dict={X: x_test, Y: new_y_test})
            test_losses.append(test_loss)
            test_accuracy.append(accuracy.eval({X: x_test, Y: new_y_test}))

        # print("pre-adversarial train accuracy is: " + str(train_accuracy[training_epochs - 1]))
        # print("pre-adversarial test accuracy is: " + str(test_accuracy[training_epochs - 1]))
        train_adversarial(sess, logits, model_fn, x_train[:adversarial_samples], new_y_train[:adversarial_samples], 10)

    # print stats
    # print("train losses = " + str(train_losses))
    # print("test losses = " + str(test_losses))
    # print("train accuracy = " + str(train_accuracy))
    # print("test accuracy = " + str(test_accuracy))


def main():
    train(mlp, 64)
    # train(conv_net, 64)


if __name__ == "__main__":
    main()
