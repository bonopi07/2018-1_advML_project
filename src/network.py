import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers


def inference_LSTM(x, x_seq_length, max_seq_length, data_dim, hidden_size, num_rnn_layers,
                   output_size, num_words, prob=1.0, train_flag=False):
    # x_one_hot = tf.one_hot(x, data_dim, name='x_one_hot')  # (batch size, seq_length, data_dim)
    word_embedding = tf.get_variable('word_embedding', [num_words, data_dim])
    x_embedding_vector = tf.nn.embedding_lookup(word_embedding, x)  # [input word #, embedding size]

    cell = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True,
                             activation=tf.tanh)
    cells = [cell for _ in range(num_rnn_layers)]
    cells = rnn.MultiRNNCell(cells)
    # cell = rnn.DropoutWrapper(cell=cell, output_keep_prob=prob)

    outputs, state = tf.nn.dynamic_rnn(cells, x_embedding_vector,
                                       sequence_length=x_seq_length,
                                       dtype=tf.float32)  # state: usually use final values
    # outputs : (?, ?, 300)
    # state   : (3, ?, 300) ?

    y_reshape = tf.reshape(outputs, [-1, max_seq_length * hidden_size])  # 3278
    # y_reshape = tf.reshape(state, [-1, 6*hidden_size])  # why 6?

    y_fc = layers.fully_connected(inputs=y_reshape,
                                  num_outputs=output_size,
                                  activation_fn=None)

    return y_fc
