import numpy as np
import LSTM
import pandas
import matplotlib.pyplot as plt
import pickle

# load the dataset
dataframe = pandas.read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')
data = np.transpose(dataset)
mx = np.max(data[0,0:100]);
labels = []
data_list = []
data_size = 144
step_size = 10
training_size = 80
plt.figure(0)
plt.plot(range(training_size), data[0, 0:training_size], 'b', label='Data')
plt.plot(range(training_size, data_size), data[0, training_size:data_size], 'r', label='Test')
plt.legend()
plt.ylabel('Passenger')
plt.xlabel('Time')
for i in range(training_size):
    labels.append(np.reshape(data[0, i+step_size], (1, 1))/mx)
    data_list.append(np.reshape(data[0, i:i+step_size], (1, step_size))/mx)
################
hidden_size = 10
input_size = 1
#################
#lstm = LSTM.BasicLSTM(hidden_size, input_size)
file = open('LSTM_Trained.txt', 'rb')
lstm = pickle.load(file)
lstm.train(data_list, labels, 50000)
test_labels = []
test_data_list =[]
mx = np.max(data[0, 100:144]);
X = np.zeros((data_size-training_size, 1))
lst = list(range(training_size-step_size, data_size-step_size))
for i in lst:
    test_labels.append(np.reshape(data[0, i+step_size], (1, 1)))
    test_data_list.append(np.reshape(data[0, i:i+step_size], (1, step_size))/mx)
for i in range(data_size-training_size):
    X[i, 0] = lstm.predict(test_data_list[i]) * mx
plt.show()
plt.figure(1)
plt.plot(range(training_size), data[0, 0:training_size], 'b', label='Data')
plt.plot(range(training_size, data_size), data[0, training_size:data_size], 'r', label='Test')
plt.plot(range(training_size, data_size), X, 'g', label='Prediction')
plt.legend()
plt.ylabel('Passenger')
plt.xlabel('Time')
plt.show()
