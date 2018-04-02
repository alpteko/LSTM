from utilis import *
from copy import deepcopy


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
            self.z = np.zeros((hidden_size, 1))
            self.i = np.zeros((hidden_size, 1))
            self.f = np.zeros((hidden_size, 1))
            self.o = np.zeros((hidden_size, 1))
            self.c = np.zeros((hidden_size, 1))
            self.h = np.zeros((hidden_size, 1))

    def calculate(self, cell, delta_h, delta_c_f):
            self.h = delta_h
            self.o = np.multiply(np.multiply(self.h, np.tanh(cell.c)), sigmoid_derivative(cell.o))
            self.c = np.multiply(np.multiply(self.h, cell.o), tanh_derivative(np.tanh(cell.c))) + delta_c_f
            self.f = np.multiply(np.multiply(self.c, cell.old_c), sigmoid_derivative(cell.f))
            self.i = np.multiply(np.multiply(self.c, cell.z), sigmoid_derivative(cell.i))
            self.z = np.multiply(np.multiply(self.c, cell.i), tanh_derivative(cell.z))


class BasicLSTM:
    def __init__(self, hidden_size, input_size):
        self.hidden_size = hidden_size
        self.input_size = input_size
        ##
        self.W_z = np.random.randn(hidden_size, input_size) * 0.1
        self.W_i = np.random.randn(hidden_size, input_size) * 0.1
        self.W_f = np.random.randn(hidden_size, input_size) * 0.1
        self.W_o = np.random.randn(hidden_size, input_size) * 0.1
        ##
        self.R_z = np.random.randn(hidden_size, hidden_size) * 0.1
        self.R_i = np.random.randn(hidden_size, hidden_size) * 0.1
        self.R_f = np.random.randn(hidden_size, hidden_size) * 0.1
        self.R_o = np.random.randn(hidden_size, hidden_size) * 0.1
        ##
        self.b_z = np.zeros((hidden_size, 1))
        self.b_i = np.zeros((hidden_size, 1))
        self.b_f = np.zeros((hidden_size, 1))
        self.b_o = np.zeros((hidden_size, 1))
        ##
        self.W_y = np.random.randn(1, hidden_size) * 0.1
        self.b_y = 0.5
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
        self.delta_W_y = np.zeros((1, hidden_size))
        self.delta_b_y = 0.0
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

    def update(self, lr, normalizer=1):
        lr /= normalizer
        self.W_z -= lr * self.delta_W_z / normalizer
        self.W_i -= lr * self.delta_W_i
        self.W_f -= lr * self.delta_W_f
        self.W_o -= lr * self.delta_W_o
        #########################################
        self.R_z -= lr * self.delta_R_z
        self.R_i -= lr * self.delta_R_i
        self.R_f -= lr * self.delta_R_f
        self.R_o -= lr * self.delta_R_o
        ##########################################
        self.b_z -= lr * self.delta_b_z
        self.b_i -= lr * self.delta_b_i
        self.b_f -= lr * self.delta_b_f
        self.b_o -= lr * self.delta_b_o
        #########################################
        self.b_y -= lr * self.delta_b_y
        self.W_y -= lr * self.delta_W_y
        ###############################################
        self.delta_W_z *= 0
        self.delta_W_i *= 0
        self.delta_W_f *= 0
        self.delta_W_o *= 0
        self.delta_R_z *= 0
        self.delta_R_i *= 0
        self.delta_R_f *= 0
        self.delta_R_o *= 0
        self.delta_b_z *= 0
        self.delta_b_i *= 0
        self.delta_b_f *= 0
        self.delta_b_o *= 0
        self.delta_b_y *= 0
        self.delta_W_y *= 0
        ###############

    def forward(self, input_stream, initial_state=0):
        self.C *= initial_state
        size = input_stream.shape
        if size[0] != self.input_size:
            print("Error: INPUT SIZE")
        h = np.zeros((self.hidden_size, 1))
        for i in range(size[1]):
            x = input_stream[:, i]
            x = x.reshape(self.input_size, 1)
            cell = self.operate(h, x)
            self.States.append(deepcopy(cell))
            h = cell.h
        return h

    def backward(self, delta_y):
        next_delta = Deltas(self.hidden_size)
        next_cell = LSTMState(self.hidden_size, self.input_size)
        self.States = deepcopy(reversed(self.States))
        for cell in self.States:
            delta = Deltas(self.hidden_size)
            delta_h = delta_y + np.dot(self.R_z.T, next_delta.z) + np.dot(self.R_i.T, next_delta.i) + np.dot(self.R_f.T, next_delta.f) + np.dot(self.R_o.T, next_delta.o)
            delta_y = 0
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
            next_delta = deepcopy(delta)
            next_cell = deepcopy(cell)
        self.C = np.zeros((self.hidden_size, 1))
        self.States = []

    def predict(self, input_stream):
        return sigmoid(np.dot(self.W_y, self.forward(input_stream)) + self.b_y)

    def test(self,input_streams, labels):
        t = len(input_streams)
        for i in range(t):
            pre = self.predict(input_streams[i])
            loss = self.error_function(pre, labels[i])
            print(input_streams[i])
            print(loss, pre, labels[i])
            self.C *= 0
        self.update(0)

    def error_function(self, prediction, label):
        #error = prediction - label
        #error = np.multiply(error, error) / 2.0
        #return error.sum(axis=0)
        return -label * np.log(prediction) - (1-label)*np.log(1-prediction)

    def derivative_h(self, prediction, label):
        return np.dot(self.W_y.T, (prediction - label))

    def train(self, data, labels, epoch, print_mode=1, lr=0.01):
        data_len = len(data)
        label_len = len(labels)
        total_loss = 0
        learning_rate = lr
        counter = 0.0;
        t = 0
        if label_len != data_len:
            print("ERROR : LABEL SIZE DOES NOT HOLD")
            exit(1)
        for i in range(10000000):
            choose = np.random.randint(0, data_len - 1)
            batch_size = data_len/10
            label = labels[choose]
            example = data[choose]
            prediction = self.predict(example)
            counter += 1.0
            delta_h = self.derivative_h(prediction, label)
            self.backward(delta_h)
            total_loss = total_loss + self.error_function(prediction, label)
            if i % batch_size == 0:
                self.update(learning_rate)
                learning_rate *= 0.99
                t += 1
                if t == epoch:
                    break
                if learning_rate < 0.001:
                    learning_rate = 0.001
                if i is not 0:
                    if print_mode == 1:
                        print("Average Error:", total_loss / batch_size, "Learning Rate:", learning_rate, t)
                        #print(prediction , label)
                        total_loss = 0
                else:
                    continue
