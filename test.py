from utilis import generate_data
from LSTM import BasicLSTM
seq_len = 8
data_size = 1000
epoch = 200
X, Y = generate_data(data_size, seq_len)
lstm = BasicLSTM(100, 1)
lstm.train(X, Y, epoch, print_mode=1, lr=0.2)
X, Y = generate_data(20, seq_len)
lstm.test(X, Y)