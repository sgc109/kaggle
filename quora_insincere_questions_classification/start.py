import os
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def read_datas():
    base_dir = os.getcwd()
    embedding_dir = os.path.join(base_dir, 'embeddings')
    glove_dir = os.path.join(embedding_dir, 'glove.840B.300d')
    glove_file = os.path.join(glove_dir, 'glove.840B.300d.txt')
    train_file = os.path.join(base_dir, 'train.csv')
    test_file = os.path.join(base_dir, 'test.csv')

    '''
        tmp = open(train_file, 'rb').read()
        for line in tmp.decode('utf-8').split('\n'):
            print(line)
    '''

    # read csv files
    train_data = pd.read_csv(train_file, quotechar='"')
    test_data = pd.read_csv(test_file, quotechar='"')

    # shuffle all rows
    train_data = train_data.sample(frac=1)

    # extract features and labels
    train_x = train_data.drop(['qid','target'], axis=1)
    train_y = train_data['target']

    # load glove
    with open(glove_file, 'r') as f:
        for line in f:
            print(line)

def main():
    read_datas()

#    train_rows, test_rows = read_datas()

if __name__ == '__main__':
    main()
