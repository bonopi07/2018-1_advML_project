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
    char2idx_file = config.get('PATH', 'CHAR_2_IDX_FILE')
    total_data = data.DataLoader(in_csv_file=data_path, char2idx_dic_file=char2idx_file)
    print('total data #:', len(total_data))

    '''
        separate data using K-fold cross validation
    '''
    data_idx = np.arange(len(total_data))
    cv = KFold(n_splits=k_fold_value, shuffle=True, random_state=0)
    for train_idx, eval_idx in cv.split(data_idx):
        '''
            separate data
        '''
        print('separate data')
        train_data = data.DataFeeder(total_data[train_idx], batch_size=batch_size, mode='train')
        eval_data = data.DataFeeder(total_data[eval_idx], batch_size=batch_size, mode='evaluate')

        print('load model')
        model_dic = {
            'epoch': int(config.get('CLASSIFIER', 'EPOCH')),
            'gpu_num': int(config.get('CLASSIFIER', 'GPU_NUM')),
            'keep_prob': float(1 - float(config.get('CLASSIFIER', 'DROPOUT_PROB'))),
            'learning_rate': float(config.get('CLASSIFIER', 'LEARNING_RATE')),
            'model_storage': config.get('CLASSIFIER', 'MODEL_STORAGE'),
            'num_classes': int(config.get('CLASSIFIER', 'NUM_CLASSES')),
            'net_type': config.get('CLASSIFIER', 'NETWORK')
        }

        classifier = model.KISNet(model_num=step,
                                  train_data=train_data,
                                  eval_data=eval_data,
                                  model_dic=model_dic)
        classifier.train()
        # classifier.evaluate()
        break
    print('*' * 50)
    pass


if __name__ == '__main__':
    run()
    pass
