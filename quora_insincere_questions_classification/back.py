import os
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def read_datas():
    base_dir = '/Users/hwangseongho/kaggle/quora_insincere_questions_classification/'
    train_dir = os.path.join(base_dir, 'train.csv')
    test_dir = os.path.join(base_dir, 'test.csv')

    train_fp = open(train_dir)
    test_fp = open(test_dir)
    
    train_rows = train_fp.read().split('\n')
    test_rows = test_fp.read().split('\n')

    train_col_names = train_rows[0]
    test_col_names = test_rows[0]

    print('train_data column names : ', train_col_names.split(','))
    print('test_data column names : ', test_col_names.split(','))

    train_rows = train_rows[1:]
    test_rows = test_rows[1:]

    ret_train_rows = [ row.split(',') for row in train_rows ]
    ret_test_rows = [ row.split(',') for row in test_rows ]

    return (ret_train_rows, ret_test_rows)



def main():
    train_rows, test_rows = read_datas()

if __name__ == '__main__':
    main()
