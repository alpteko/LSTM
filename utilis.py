import numpy as np


def sigmoid(x):
    x = np.clip( x, -500, 500 )
    return 1. / (1 + np.exp(-x))


def sigmoid_derivative(values):
    return np.multiply(values,(1-values))


def tanh_derivative(values):
    return 1. - values ** 2


def softmax(h):
    e_h = np.exp(h - np.max(h))
    return e_h / e_h.sum(axis=0)

def adding_problem_generator(seq_len):
    X_num = np.random.binomial(n=1, p=0.5, size=(1, seq_len))
    Y = 0
    if np.sum(X_num) > seq_len/2:
        Y = 1
    return X_num, Y

def generate_data(N,seq_len):
    data = []
    label = []
    for i in range(N):
        X, Y = adding_problem_generator(seq_len)
        data.append(X)
        label.append(Y)
    return data,label