import mnist # pip install mnist
import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork
from activations import relu, d_relu, softmax, d_softmax
from loss_functions import categorical_crossentropy, d_categorical_crossentropy
from optimizers import SGD

x_train, y_train = mnist.train_images(), mnist.train_labels()

n_features = x_train.shape[1] * x_train.shape[2]

x_train_flatten = x_train.reshape((x_train.shape[0], n_features)).astype(np.float64)

x_test, y_test = mnist.test_images(), mnist.test_labels()

x_test_flatten = x_test.reshape((x_test.shape[0], n_features)).astype(np.float64)

nn_mnist = NeuralNetwork(
    layers=[n_features, 100, 10],
    hidden_activation=(relu, d_relu),
    output_activation=(softmax, d_softmax),
    loss=(categorical_crossentropy, d_categorical_crossentropy),
    optimizer=SGD()
)

mnist_loss_hist = nn_mnist.fit(
    x=x_train_flatten,
    y=y_train,
    batch_size=512,
    epochs=100,
    categorical=True
)

plt.figure(figsize=[10, 8], dpi=120)
plt.plot(mnist_loss_hist)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
# plt.savefig('imgs/MNIST_loss.png')

print('Train: '+nn_mnist.score(x_train_flatten, y_train, accuracy=True))

print('Test: '+nn_mnist.score(x_test_flatten, y_test, accuracy=True))