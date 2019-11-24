import matplotlib.pyplot as plt

training_epochs = 150


def plot_losses(train_losses, test_losses, epochs):
    plt.subplot(121)
    plt.plot(range(epochs), train_losses, label="train_loss")
    plt.plot(range(epochs), test_losses, label="test_loss")
    plt.xlabel("epochs")
    plt.ylabel("losses")
    plt.title("losses and epochs")
    plt.legend()
    # plt.savefig("graphs_new/losses.png", format='png')


def plot_accuracy(train_acc, test_acc, epochs):
    plt.subplot(122)
    plt.plot(range(epochs), train_acc, label='train_acc')
    plt.plot(range(epochs), test_acc, label='test_acc')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title('accuracy and epochs')
    plt.legend()
    # plt.savefig("graphs_new/accuracy.png", format='png')


def plot_graphs(train_losses, test_losses, train_accuracy, test_accuracy, epochs):
    plot_losses(train_losses, test_losses, epochs)
    plot_accuracy(train_accuracy, test_accuracy, epochs)
    plt.show()

