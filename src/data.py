import csv
import numpy as np
import pickle
import random


class DataLoader:
    '''
        Class: DataLoader
        Role: store whole data, related file(e.g. char2idx dictionary)
    '''
    def __init__(self, in_csv_file, unit2idx_dic_file):
        self.class_description = 'hotel review rating prediction using LSTM'

        '''
            initialize member variable
        '''
        self.in_csv_file = in_csv_file
        self.unit2idx_dic_file = unit2idx_dic_file

        self.str_data_list = list()
        self.idx_data_list = list()
        self.label_list = list()

        self.number_of_data = 0
        self.number_of_words = 0

        '''
            allocate all data into memory
        '''
        self._load_data()
        pass

    def _load_data(self):
        # load file
        with open(self.unit2idx_dic_file, 'rb') as f:
            unit_dic = pickle.load(f)
            self.number_of_words = len(unit_dic)

        data_cnt = 0
        with open(self.in_csv_file, 'r', encoding='utf-8') as f:
            rdr = csv.reader(f)
            for line in rdr:
                data_cnt += 1

                rating, review = int(line[0]), line[1]

                self.str_data_list.append(review)
                # self.idx_data_list.append([unit_dic[c] for c in review])  # char2idx
                self.idx_data_list.append([unit_dic[w] for w in review.split(' ')])  # word2idx
                self.label_list.append(rating)

        # set max sequence length
        # self.max_seq_len = max([len(seq) for seq in self.str_data_list])
        self.max_seq_len = max([len(seq.split(' ')) for seq in self.str_data_list])

        # count data
        self.number_of_data = data_cnt

        # casting list to numpy
        self.idx_data_list = np.array(self.idx_data_list)
        self.label_list = np.array(self.label_list)
        pass

    def get_max_seq_length(self):
        return self.max_seq_len

    def get_number_of_words(self):
        return self.number_of_words

    def __len__(self):
        return self.number_of_data

    def __getitem__(self, item):
        if isinstance(item, int):
            return [self.idx_data_list[item], self.label_list[item]]
        elif type(item).__module__ == np.__name__:
            return [self.idx_data_list[item], self.label_list[item]]
        else:
            raise NotImplementedError


class DataFeeder:
    '''
        Class: DataFeeder
        Role
            - store specific data, label
            - feed data to model
    '''
    def __init__(self, data_label_list, max_seq_len, batch_size, mode):
        self.iter_mode = mode  # for mini-batch data feeding

        '''
            initialize member variable
        '''
        self.data_list = data_label_list[0]
        self.label_list = data_label_list[1]
        self.data_len_list = list()

        self.number_of_data = len(self.data_list)

        '''
            set batch size
        '''
        self.batch_size = batch_size

        '''
            set max sequence length
        '''
        self.max_seq_len = max_seq_len

        '''
            zero padding
        '''
        for idx in range(self.number_of_data):
            data_len = len(self.data_list[idx])
            self.data_len_list.append(data_len)
            if self.max_seq_len > data_len:
                self.data_list[idx] += (self.max_seq_len - data_len) * [0]
        pass

    def get_batch(self):
        return self.batch_size

    def __len__(self):
        return self.number_of_data

    def __iter__(self):
        if self.iter_mode == 'train':  # mini-batch
            '''
                initialize batch data/label
            '''
            batch_data_lists, batch_label_lists = list(), list()
            batch_data_len_lists = list()

            while True:
                batch_data_lists.clear()
                batch_label_lists.clear()
                batch_data_len_lists.clear()

                '''
                    shuffle index list
                '''
                idx_list = np.arange(self.number_of_data)
                random.shuffle(idx_list)

                '''
                    create batch file/label list
                '''
                for idx in idx_list[:self.batch_size]:
                    batch_data_lists.append(self.data_list[idx])
                    batch_label_lists.append(self.label_list[idx])
                    batch_data_len_lists.append(self.data_len_list[idx])

                yield (batch_data_lists, batch_label_lists, batch_data_len_lists)
        else:  # evaluation mode
            for data, label, data_len in zip(self.data_list, self.label_list, self.data_len_list):
                yield [data], label, [data_len]