import numpy as np

EPS = np.finfo(np.float64).eps

def identity(x):
    return x

def d_identity(x):
    return np.tile(np.identity(x.shape[1]), (x.shape[0], 1, 1))

def relu(x):
    return np.maximum(x, 0)

def d_relu(x):
    return np.vectorize(lambda v: 1 if v > 0 else 0)(x)

def softmax(x):
    x = x - x.max(axis=1).reshape((-1, 1))
    exp = np.exp(x)
    s = np.sum(exp, axis=1).reshape((-1, 1))
    return exp/(s+EPS)

def d_softmax(x):
    s = softmax(x)
    D = np.stack([np.diag(s[i, :]) for i in range(s.shape[0])], axis=0)
    comb = np.matmul(np.expand_dims(s, 2), np.expand_dims(s, 1))
    return D - comb