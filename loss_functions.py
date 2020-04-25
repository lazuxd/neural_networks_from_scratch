import numpy as np

EPS = np.finfo(np.float64).eps

def mean_squared_error(y_pred, y_true):
    return np.mean((y_pred-y_true)**2, axis=1).reshape((-1, 1))

def d_mean_squared_error(y_pred, y_true):
    return np.expand_dims((2/y_pred.shape[1])*(y_pred-y_true), 1)

def categorical_crossentropy(y_pred, y_true):
    return -np.log(np.sum(y_true*y_pred, axis=1)+EPS)

def d_categorical_crossentropy(y_pred, y_true):
    return np.expand_dims(-y_true/(y_pred+EPS), 1)