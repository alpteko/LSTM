import numpy as np
import LSTM
import pandas
import matplotlib.pyplot as plt

# load the dataset
dataframe = pandas.read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')
data = np.transpose(dataset)
mx = np.max(data[0,0:100]);
labels = []
data_list = []
for i in range(100):
    labels.append(np.reshape(data[0, i+4],(1,1))/mx)
    data_list.append(np.reshape(data[0, i:i+4],(1,4))/mx)
################
hidden_size = 10;
input_size = 1;
#################
lstm = LSTM.BasicLSTM(hidden_size, input_size)
lstm.train(data_list, labels, 10000)
test_labels = []
test_data_list =[]
mx = np.max(data[0,100:144]);
lst = list(range(100,140))
for i in lst:
    test_labels.append(np.reshape(data[0, i+4], (1, 1)))
    test_data_list.append(np.reshape(data[0, i:i+4], (1, 4))/mx)
X = np.zeros((40,1))
for i in range(40):
    X[i, 0] = lstm.predict(test_data_list[i])*mx
plt.plot(data[0,104:144],'b')
plt.plot(X,'g')
plt.show()

