import pickle
import numpy as np
import os

def load_CIFAR_batch(filename):
    """load single batch of CIFAR"""
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1') # encoding ='latin1' it is to remove python2/3 compatibility issue
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y