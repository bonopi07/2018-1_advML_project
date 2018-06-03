import configparser
from tensorflow.contrib import rnn
import tensorflow as tf

# config parameters
config = configparser.ConfigParser()
config.read('config.ini')

# define parameters
hidden_size = 10
num_rnn_layers = 3


def inference_LSTM(x, prob=1.0, train_flag=False):
    cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
    cell = rnn.MultiRNNCell([cell]*num_rnn_layers, state_is_tuple=True)
    cell = rnn.DropoutWrapper(cell=cell, output_keep_prob=prob)

    # loss function:
    outputs, _ = tf.nn.dynamic_rnn(cell, x, sequence_length=[],
                                   dtype=tf.float32)

    return cell
