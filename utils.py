import numpy as np

def initw(n, d):
    magic_number = 0.1
    return (np.random.rand(n, d) * 2 - 1) * magic_number

def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

