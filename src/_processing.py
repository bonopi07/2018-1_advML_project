import csv
import os
import re


# function: 입력 경로에 대한 모든 파일 경로를 리스트로 반환하는 함수
def walk_dir(input_path):
    result = list()
    for path, _, files in os.walk(input_path):
        for file in files:
            file_path = os.path.join(path, file)  # store "file path"
            result.append(file_path)
    return result


# function: 주어진 csv file에 대해 특정 정규표현식을 만족하는 패턴만 추출하여 csv를 저장하는 함수
def parse_qualified_data(in_file_name, out_file_name,
                         in_encoding_type='', out_encoding_type='', rule=''):
    if in_encoding_type == '':
        in_encoding_type = 'latin-1'
    if out_encoding_type == '':
        out_encoding_type = 'utf-8'
    if rule == '':
        rule = '[a-zA-Z0-9\s.,_/?!\-]+'  # 알파벳, 숫자, 공백, 특정 특수문자

    whole_info = list()

    print('read')
    with open(in_file_name, 'r', encoding=in_encoding_type) as f:
        rdr = csv.reader(f)
        cnt = 0
        for line in rdr:
            cnt += 1

            rating, text = int(line[0])-1, line[1]

            regex = re.compile(rule)
            mod_sentence = ''.join(regex.findall(text)).lower()

            whole_info.append([rating, mod_sentence])

    print('write')
    with open(out_file_name, 'w', encoding=out_encoding_type, newline='') as f:
        wr = csv.writer(f)
        for line in whole_info:
            wr.writerow(line)
    pass


# function:
def create_char_2_idx_dic(in_file_name, pickle_file_name='char2idx.pickle'):
    # csv file 에서 sentence 에 해당하는 data 수집
    sentences = list()
    with open(in_file_name, 'r', encoding='utf-8') as f:
        rdr = csv.reader(f)
        for line in rdr:
            sentences.append(line[1])

    # char set 생성
    char_set = set()
    for sentence in sentences:
        char_set.update([char for char in sentence])
    print('result char set:', char_set)
    print('result char set size:', len(char_set))

    # char dict 생성
    char_dic = {w: i for i, w in enumerate(char_set)}
    print('result char dict:', char_dic)
    print('result char dict size:', len(char_dic))

    # unicode 에서의 공백을 ascii code 일 때와 병합
    char_dic['\xa0'] = char_dic[' ']  # \x85는 unicode로 newline

    # pickle format 으로 저장
    # with open(pickle_file_name, 'wb') as f:
    #     pickle.dump(char_dic, f, protocol=pickle.HIGHEST_PROTOCOL)
    pass


if __name__ == '__main__':
    '''
        parse qualified data
    '''
    parse_qualified_data(in_file_name='../data/data_rating_text.csv',
                         out_file_name='../data/data_final.csv',
                         in_encoding_type='latin-1',
                         out_encoding_type='utf-8',
                         rule='[a-zA-Z0-9\s.,_/?!\-]+')

    '''
        create char2idx dictionary
    '''
    create_char_2_idx_dic(in_file_name='../data/data_final.csv',
                          pickle_file_name='../data/char2idx.pickle')

    pass
