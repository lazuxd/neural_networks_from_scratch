import numpy as np

def to_categorical(labels):
    n_classes = labels.max()+1
    y = np.zeros((labels.shape[0], n_classes))
    
    for i in range(labels.shape[0]):
        y[i, labels[i]] = 1
    
    return y