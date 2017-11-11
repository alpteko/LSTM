import numpy as np
import LSTM
lstm = LSTM.BasicLSTM(3,3)
lstm.forward(np.random.rand(3, 10))
lstm.backward(0.1)
print(lstm.delta_b_z)
