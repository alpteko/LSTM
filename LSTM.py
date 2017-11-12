import random
import numpy as np
import math
import tensorflow as tf
import pickle

def sigmoid(x):
    x = np.clip( x, -500, 500 )
    return 1. / (1 + np.exp(-x))


def sigmoid_derivative(values):
    return values*(1-values)


def tanh_derivative(values):
    return 1. - values ** 2


def softmax(h):
    e_h = np.exp(h - np.max(h))
    return e_h / e_h.sum(axis=0)


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
    def __init__(self,hidden_size, input_size):
        self.hidden_size = hidden_size
        self.input_size = input_size
        ### Matricies
        self.W_z = 2*np.random.rand(hidden_size, input_size)-1
        self.W_i = 2*np.random.rand(hidden_size, input_size)-1
        self.W_f = 2*np.random.rand(hidden_size, input_size)-1
        self.W_o = 2*np.random.rand(hidden_size, input_size)-1
        ##
        self.R_z = 2*np.random.rand(hidden_size, hidden_size)-1
        self.R_i = 2*np.random.rand(hidden_size, hidden_size)-1
        self.R_f = 2*np.random.rand(hidden_size, hidden_size)-1
        self.R_o = 2*np.random.rand(hidden_size, hidden_size)-1
        ##
        self.b_z = 2*np.random.rand(hidden_size, 1)-1
        self.b_i = 2*np.random.rand(hidden_size, 1)-1
        self.b_f = 2*np.random.rand(hidden_size, 1)-1
        self.b_o = 2*np.random.rand(hidden_size, 1)-1
        ##
        self.W_y = np.random.randn(1, hidden_size)
        self.b_y = 0.0;
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
        self.delta_W_y = np.zeros((1,hidden_size))
        self.delta_b_y = 0.0;
        ##
        self.C = np.zeros((hidden_size, 1))
        self.States = []

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
        lr = lr * -1;
        self.W_z = self.W_z + lr * self.delta_W_z
        self.W_i = self.W_i + lr * self.delta_W_i
        self.W_f = self.W_f + lr * self.delta_W_f
        self.W_o = self.W_o + lr * self.delta_W_o
        #########################################
        self.R_z = self.R_z + lr * self.delta_R_z
        self.R_i = self.R_i + lr * self.delta_R_i
        self.R_f = self.R_f + lr * self.delta_R_f
        self.R_o = self.R_o + lr * self.delta_R_o
        ##########################################
        self.b_z = self.b_z + lr * self.delta_b_z
        self.b_i = self.b_i + lr * self.delta_b_i
        self.b_f = self.b_f + lr * self.delta_b_f
        self.b_o = self.b_o + lr * self.delta_b_o
        #########################################
        self.b_y += self.delta_b_y
        self.W_y += self.delta_W_y
        ###############################################
        self.delta_W_z = np.zeros((self.hidden_size, self.input_size))
        self.delta_W_i = np.zeros((self.hidden_size, self.input_size))
        self.delta_W_f = np.zeros((self.hidden_size, self.input_size))
        self.delta_W_o = np.zeros((self.hidden_size, self.input_size))
        self.delta_R_z = np.zeros((self.hidden_size, self.hidden_size))
        self.delta_R_i = np.zeros((self.hidden_size, self.hidden_size))
        self.delta_R_f = np.zeros((self.hidden_size, self.hidden_size))
        self.delta_R_o = np.zeros((self.hidden_size, self.hidden_size))
        self.delta_b_z = np.zeros((self.hidden_size, 1))
        self.delta_b_i = np.zeros((self.hidden_size, 1))
        self.delta_b_f = np.zeros((self.hidden_size, 1))
        self.delta_b_o = np.zeros((self.hidden_size, 1))
        self.delta_b_y *= 0
        self.delta_W_y *= 0
        ###############

        ##
        self.C = np.zeros((self.hidden_size, 1))
        self.States = []


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
        return h

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

    def predict(self, input_stream):
            return np.dot(self.W_y,self.forward(input_stream)+self.b_y)
            estimate_y = softmax(self.forward(input_stream))
            return estimate_y

    def error_function(self, prediction, label):
            error = prediction - label;
            error = np.multiply(error, error)/2.0
            return error.sum(axis=0)
            error = np.multiply(label, np.log(prediction))

    def derivative_h(self,prediction, label):
            return np.dot(self.W_y.T,(prediction - label))

    def train(self, data, labels, iteration_number):
        data_len = len(data)
        label_len = len(labels)
        total_loss = 0
        learning_rate = 0.1
        counter = 0.0;
        i = 0;
        meta_rate = 1.1
        if label_len != data_len:
            print("ERROR : LABEL SIZE DOES NOT HOLD")
            exit(1)
        while(1):
            i +=1;
            choose = np.random.randint(0, data_len-1)
            label = labels[choose]
            example = data[choose]
            prediction = self.predict(example)
            counter += 1.0
            delta_h = self.derivative_h(prediction, label)
            self.backward(delta_h)
            total_loss = total_loss + self.error_function(prediction, label)
            self.update(learning_rate)
            if i % 1000 == 0:
                print("Average Error:", total_loss/counter, "Learning Rate:", learning_rate)
            learning_rate *= 0.99999
            if total_loss/counter < 0.01:
                break







