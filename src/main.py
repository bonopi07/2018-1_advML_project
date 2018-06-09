import configparser
import data
import model
import numpy as np
from sklearn.model_selection import KFold

# config parameters
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read('config.ini')


def run():
    print('*' * 50)

    '''
        define initial parameters
    '''
    batch_size = int(config.get('CLASSIFIER', 'BATCH_SIZE'))
    k_fold_value = int(config.get('BASIC_INFO', 'K_FOLD_VALUE'))
    step = int(config.get('BASIC_INFO', 'MODEL_STEP'))  # deep learning step number

    '''
        load data
    '''
    print('step {}'.format(step))
    print('load data')

    data_path = config.get('PATH', 'BASE_DATA_CSV')
    # char2idx_file = config.get('PATH', 'CHAR_2_IDX_FILE')
    word2idx_file = config.get('PATH', 'WORD_2_IDX_FILE')
    total_data = data.DataLoader(in_csv_file=data_path, unit2idx_dic_file=word2idx_file)

    max_seq_length = total_data.get_max_seq_length()
    number_of_words = total_data.get_number_of_words()
    print('max seq length #:', max_seq_length)
    print('input words #:', number_of_words)
    print('total data #:', len(total_data))

    '''
        separate data using K-fold cross validation
    '''
    accuracy_list = list()
    data_idx = np.arange(len(total_data))
    cv = KFold(n_splits=k_fold_value, shuffle=True, random_state=0)
    for exp_idx, (train_idx, eval_idx) in enumerate(cv.split(data_idx)):
        '''
            separate data
        '''
        print('separate data {}'.format(exp_idx))
        train_data = data.DataFeeder(total_data[train_idx], max_seq_len=max_seq_length,
                                     batch_size=batch_size, mode='train')
        eval_data = data.DataFeeder(total_data[eval_idx], max_seq_len=max_seq_length,
                                    batch_size=batch_size, mode='evaluate')

        print('load model')
        model_dic = {
            'dropout_prob': float(config.get('CLASSIFIER', 'DROPOUT_PROB')),
            'epoch': int(config.get('CLASSIFIER', 'EPOCH')),
            'gpu_num': int(config.get('CLASSIFIER', 'GPU_NUM')),
            'hidden_size': int(config.get('CLASSIFIER', 'HIDDEN_SIZE')),
            'learning_rate': float(config.get('CLASSIFIER', 'LEARNING_RATE')),
            'max_seq_length': max_seq_length,
            'model_storage': config.get('CLASSIFIER', 'MODEL_STORAGE'),
            'network': config.get('CLASSIFIER', 'NETWORK'),
            'num_rnn_layers': int(config.get('CLASSIFIER', 'NUM_RNN_LAYERS')),
            'num_words': number_of_words,
            'output_size': int(config.get('CLASSIFIER', 'OUTPUT_SIZE'))
        }

        classifier = model.KISNet(model_num=step,
                                  train_data=train_data,
                                  eval_data=eval_data,
                                  model_dic=model_dic)
        classifier.train()
        accuracy = classifier.evaluate()

        accuracy_list.append(accuracy)
    print('*' * 50)
    print('average accuracy:', (max(accuracy_list)/len(accuracy_list)))
    pass


if __name__ == '__main__':
    run()
    pass
