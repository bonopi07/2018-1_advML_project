import network as net
import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib import seq2seq
import time


class KISNet:
    def __init__(self, model_num, train_data, eval_data, model_dic):
        '''
            init deep learning environment variable
        '''
        self.data_dim = model_dic['hidden_size']
        self.dropout_prob = model_dic['dropout_prob']
        self.epoch = model_dic['epoch']
        self.gpu_num = model_dic['gpu_num']
        self.hidden_size = model_dic['hidden_size']
        self.learning_rate = model_dic['learning_rate']
        self.max_seq_length = model_dic['max_seq_length']
        self.model_num = model_num
        self.model_storage = model_dic['model_storage']
        self.network_type = model_dic['network']
        self.num_rnn_layers = model_dic['num_rnn_layers']
        self.num_words = model_dic['num_words']
        self.output_size = model_dic['output_size']

        self.train_flag = False

        self.batch_size = train_data.get_batch()

        '''
            init data
        '''
        self.train_data = train_data
        self.eval_data = eval_data
        pass

    def get_model_snapshot_path(self):
        # create model storage
        model_storage = self.model_storage + str(self.model_num)
        if not os.path.isdir(model_storage):
            os.makedirs(model_storage)

        return os.path.normpath(os.path.abspath('./{}/model.ckpt'.format(model_storage)))

    def inference(self, x, x_seq_length, prob=1.0):
        if self.network_type == 'LSTM':
            return net.inference_LSTM(x, x_seq_length, self.max_seq_length, self.data_dim, self.hidden_size,
                                      self.num_rnn_layers, self.output_size, self.num_words, prob, self.train_flag)
        else:
            raise NotImplementedError
        pass

    def train(self):
        print('training start')
        self.train_flag = True

        # design network architecture
        with tf.device('/gpu:{}'.format(self.gpu_num)):
            tf.reset_default_graph()

            x = tf.placeholder(tf.int32, shape=[None, None], name='x')
            x_seq_length = tf.placeholder(tf.int32, shape=[None], name='x_seq_length')

            y = tf.placeholder(tf.int32, shape=[None], name='y')
            y_one_hot = tf.one_hot(y, self.output_size, name='y_one_hot')

            prob = tf.placeholder(tf.float32, name='prob')

            y_ = self.inference(x, x_seq_length, prob)

            # loss function 1
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_, labels=y_one_hot))

            # loss function 2
            # weight = tf.ones([self.train_batch_size, max_seq_length])
            # sequence_loss = seq2seq.sequence_loss(logits=y_, targets=y,
            #                                       weights=weight)
            # mean_loss = tf.reduce_mean(sequence_loss)

            # optimizer: Adaptive momentum optimizer
            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)

            # predict
            prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y_one_hot, 1))
            accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

        # create model snapshot
        model_path = self.get_model_snapshot_path()

        # training session start
        keep_prob = float(1 - self.dropout_prob)
        model_saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True

        with tf.Session(config=tf_config) as sess:
            sess.run(init)

            number_of_data = len(self.train_data)
            print('file # : {}'.format(number_of_data))

            # train_epoch * number_of_data = batch_size * iteration
            train_time = time.time()
            total_iteration = (self.epoch*number_of_data)//self.batch_size
            for iteration, (train_data, train_label, train_data_len) in enumerate(self.train_data):
                if iteration >= total_iteration:
                    break

                _cost, _, acc = sess.run([cost, optimizer, accuracy],
                                         feed_dict={x: train_data, y: train_label,
                                                    x_seq_length: train_data_len, prob: keep_prob})
                if iteration % 50 == 0:
                    print('[{i}/{total}] cost: {cost:.2f} / acc: {acc:.2f} / elapsed time: {time:.2f}'.format(
                        i=iteration, total=total_iteration, cost=_cost, acc=acc, time=time.time()-train_time
                    ))
                if iteration % 100 == 0:
                    model_saver.save(sess, model_path)

                iteration += 1
            train_time = time.time() - train_time
            model_saver.save(sess, model_path)
        print('training time : {}'.format(train_time))
        print('------training finish------')
        pass

    def evaluate(self):
        print('evaluating start')
        self.train_flag = False

        # design network architecture
        with tf.device('/gpu:{}'.format(self.gpu_num)):
            tf.reset_default_graph()

            x = tf.placeholder(tf.int32, shape=[None, None], name='x')
            x_seq_length = tf.placeholder(tf.int32, shape=[None], name='x_seq_length')

            y_ = self.inference(x, x_seq_length)

        # restore model snapshot
        model_path = self.get_model_snapshot_path()

        # evaluating session start
        model_saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        tf_config = tf.ConfigProto(allow_soft_placement=True)

        with tf.Session(config=tf_config) as sess:
            sess.run(init)

            number_of_data = len(self.eval_data)
            print('file # : {}'.format(number_of_data))

            model_saver.restore(sess, model_path)

            answer_cnt = 0
            actual_labels, pred_labels = list(), list()

            eval_time = time.time()
            for iteration, (eval_data, eval_label, eval_data_len) in enumerate(self.eval_data):
                try:
                    pred = sess.run(y_, feed_dict={x: eval_data, x_seq_length: eval_data_len})

                    pred_label = np.array(pred).reshape(-1).argmax(-1)  # scalar value
                    actual_label = eval_label

                    if pred_label == actual_label:
                        answer_cnt += 1
                    if iteration % 1000 == 0:
                        print('[{i}/{total}] acc: {acc:.2f} / elapsed time: {time:.2f}'.format(
                            i=iteration, total=number_of_data, acc=(answer_cnt/(iteration+1)), time=time.time()-eval_time
                        ))
                except Exception as e:
                    print(e)
                    pred_label = -1
                    actual_label = -1
                pred_labels.append(pred_label)
                actual_labels.append(actual_label)
            eval_time = time.time() - eval_time
        total_accuracy = float(100. * (answer_cnt / number_of_data))
        print('test time : {}'.format(eval_time))
        print('accuracy : {}'.format(total_accuracy))
        print('-----evaluating finish-----')

        # plot confusion matrix
        # with open('result{}.pickle'.format(self.model_num), 'wb') as f:
        #     _pickle.dump(file_lists, f)
        #     _pickle.dump(actual_labels, f)
        #     _pickle.dump(pred_labels, f)
        #
        # analysis.plot_confusion_matrix(actual_labels, pred_labels)

        return total_accuracy
