import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork
from activations import relu, d_relu, identity, d_identity
from loss_functions import mean_squared_error, d_mean_squared_error
from optimizers import SGD

x, y = fetch_california_housing(return_X_y=True)
y = y.reshape((-1, 1)) # Turn it into column vector

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

x_mean = np.mean(x_train, 0)
x_std_dev = np.std(x_train, 0)

x_train = (x_train-x_mean)/x_std_dev # Normalize input data

n_features = x_train.shape[1]

nn_housing = NeuralNetwork(
    layers=[n_features, 100, 20, 1],
    hidden_activation=(relu, d_relu),
    output_activation=(identity, d_identity),
    loss=(mean_squared_error, d_mean_squared_error),
    optimizer=SGD()
)

housing_loss_hist = nn_housing.fit(
    x=x_train,
    y=y_train,
    batch_size=512,
    epochs=100
)

plt.figure(figsize=[10, 8], dpi=120)
plt.plot(housing_loss_hist)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
# plt.savefig('imgs/Housing_loss.png')

print('Train: '+nn_housing.score(x_train, y_train))

x_test = (x_test-x_mean)/x_std_dev # Normalize testing data
print('Test: '+nn_housing.score(x_test, y_test))