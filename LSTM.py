import random
import numpy as np
import math


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def sigmoid_derivative(values):
    return values*(1-values)


def tanh_derivative(values):
    return 1. - values ** 2


class LSTMState:
    def __init__(self, hidden_size, input_size):
        self.z = np.zeros((hidden_size, 1))
        self.i = np.zeros((hidden_size, 1))
        self.f = np.zeros((hidden_size, 1))
        self.o = np.zeros((hidden_size, 1))
        self.c = np.zeros((hidden_size, 1))
        self.h = np.zeros((hidden_size, 1))
        self.old_h = np.zeros((hidden_size, 1))
        self.old_c = np.zeros((hidden_size, 1))
        self.x = np.zeros((input_size, 1))


class Deltas:
    def __init__(self, hidden_size):
        self.z = np.zeros(hidden_size)
        self.i = np.zeros(hidden_size)
        self.f = np.zeros(hidden_size)
        self.o = np.zeros(hidden_size)
        self.c = np.zeros(hidden_size)
        self.h = np.zeros(hidden_size)

    def calculate(self, cell, delta_h, delta_c_f):
        self.h = delta_h
        self.o = np.multiply(np.multiply(self.h,np.tanh(cell.c)),sigmoid_derivative(cell.o))
        self.c = np.multiply(np.multiply(self.h, cell.o),tanh_derivative(cell.c))
        self.f = np.multiply(np.multiply(self.c, cell.old_c), sigmoid_derivative(cell.f)) + delta_c_f
        self.i = np.multiply(np.multiply(self.c, cell.z), sigmoid_derivative(cell.i))
        self.z = np.multiply(np.multiply(self.c, cell.i), sigmoid_derivative(cell.z))


class BasicLSTM:
    def __init__(self,hidden_size, input_size, lr):
        self.hidden_size = hidden_size
        self.input_size = input_size
        ### Matricies
        self.W_z = np.random.randn(hidden_size, input_size)
        self.W_i = np.random.randn(hidden_size, input_size)
        self.W_f = np.random.randn(hidden_size, input_size)
        self.W_o = np.random.randn(hidden_size, input_size)
        ##
        self.R_z = np.random.randn(hidden_size, hidden_size)
        self.R_i = np.random.randn(hidden_size, hidden_size)
        self.R_f = np.random.randn(hidden_size, hidden_size)
        self.R_o = np.random.randn(hidden_size, hidden_size)
        ##
        self.b_z = np.random.randn(hidden_size, 1)
        self.b_i = np.random.randn(hidden_size, 1)
        self.b_f = np.random.randn(hidden_size, 1)
        self.b_o = np.random.randn(hidden_size, 1)
        ##
        self.delta_W_z = np.zeros((hidden_size, input_size))
        self.delta_W_i = np.zeros((hidden_size, input_size))
        self.delta_W_f = np.zeros((hidden_size, input_size))
        self.delta_W_o = np.zeros((hidden_size, input_size))
        self.delta_R_z = np.zeros((hidden_size, hidden_size))
        self.delta_R_i = np.zeros((hidden_size, hidden_size))
        self.delta_R_f = np.zeros((hidden_size, hidden_size))
        self.delta_R_o = np.zeros((hidden_size, hidden_size))
        self.delta_b_z = np.zeros((hidden_size, 1))
        self.delta_b_i = np.zeros((hidden_size, 1))
        self.delta_b_f = np.zeros((hidden_size, 1))
        self.delta_b_o = np.zeros((hidden_size, 1))
        ##
        self.C = np.zeros((hidden_size, 1))
        self.States = []
        self.lr = lr

    def operate(self, old_h, x):
        cell = LSTMState(self.hidden_size, self.input_size)
        cell.old_h = old_h
        cell.z = np.tanh(np.dot(self.W_z, x) + np.dot(self.R_z, old_h) + self.b_z)
        cell.i = sigmoid(np.dot(self.W_i, x) + np.dot(self.R_i, old_h) + self.b_i)
        cell.f = sigmoid(np.dot(self.W_f, x) + np.dot(self.R_f, old_h) + self.b_f)
        cell.o = sigmoid(np.dot(self.W_o, x) + np.dot(self.R_o, old_h) + self.b_o)
        cell.old_c = self.C
        self.C = np.multiply(cell.z, cell.i) + np.multiply(cell.old_c, cell.f)
        cell.c = self.C
        cell.h = np.multiply(np.tanh(cell.c), cell.o)
        cell.x = x
        return cell

    def update(self, lr):
        self.W_z = self.W_z + lr * self.delta_W_z
        self.W_i = self.W_i + lr * self.delta_W_i
        self.W_f = self.W_f + lr * self.delta_W_f
        self.W_o = self.W_o + lr * self.delta_W_o
        ##############################################
        self.R_z = self.R_z + lr * self.delta_R_z
        self.R_i = self.R_i + lr * self.delta_R_i
        self.R_f = self.R_f + lr * self.delta_R_f
        self.R_o = self.R_o + lr * self.delta_R_o
        ##############################################
        self.b_z = self.b_z + lr * self.delta_b_z
        self.b_i = self.b_i + lr * self.delta_b_i
        self.b_f = self.b_f + lr * self.delta_b_f
        self.b_o = self.b_o + lr * self.delta_b_o

    def forward(self, input_stream):
        size = input_stream.shape
        if size[0] != self.input_size:
            print("Error: INPUT SIZE")
        h = np.zeros((self.hidden_size, 1))
        for i in range(size[1]):
            x = input_stream[:, i]
            x = x.reshape(self.input_size,1)
            cell = self.operate(h, x)
            self.States.append(cell)
            h = cell.h

    def backward(self, delta_h):
        n_step = len(self.States)
        next_delta = Deltas(self.hidden_size)
        next_cell = LSTMState(self.hidden_size,self.input_size)
        for i in range(n_step-1,-1,-1):
            cell = self.States[i]
            delta = Deltas(self.hidden_size)
            if i == n_step-1:
                delta.calculate(cell, delta_h, 0)
            else:
                delta_h = np.dot(np.transpose(self.W_z), next_delta.z) + np.dot(np.transpose(self.W_i), next_delta.f) + np.dot(np.transpose(self.W_f), next_delta.f) + np.dot(np.transpose(self.W_o), next_delta.o)
                delta_c_f = np.multiply(next_delta.c, next_cell.f)
                delta.calculate(cell, delta_h, delta_c_f)
            self.delta_W_z = self.delta_W_z + np.outer(delta.z, cell.x)
            self.delta_R_z = self.delta_R_z + np.outer(delta.z, cell.old_h)
            self.delta_b_z = self.delta_b_z + delta.z
            ########
            self.delta_W_i = self.delta_W_i + np.outer(delta.i, cell.x)
            self.delta_R_i = self.delta_R_i + np.outer(delta.i, cell.old_h)
            self.delta_b_i = self.delta_b_i + delta.i
            ########
            self.delta_W_f = self.delta_W_f + np.outer(delta.f, cell.x)
            self.delta_R_f = self.delta_R_f + np.outer(delta.f, cell.old_h)
            self.delta_b_f = self.delta_b_f + delta.f
            ###########
            self.delta_W_o = self.delta_W_o + np.outer(delta.o, cell.x)
            self.delta_R_o = self.delta_R_o + np.outer(delta.o, cell.old_h)
            self.delta_b_o = self.delta_b_o + delta.o
            ###########
            next_delta = delta
            next_cell = cell






