import os

def check_exist_middle_new_line(rows):
    for s in rows:
        if len(s) < 2 or s[0] != '\"' or s[-1] != '\"':
            return False
    return True

def read_datas():
    base_dir = '/Users/hwangseongho/kaggle/'
    train_dir = os.path.join(base_dir, 'train.csv')
    test_dir = os.path.join(base_dir, 'test.csv')

    train_fp = open(train_dir)
    test_fp = open(test_dir)
    
    train_rows = train_fp.read().split('\n')
    test_rows = test_fp.read().split('\n')

    train_col_names = train_rows[0]
    test_col_names = test_rows[0]

    print('train_data column names : ', train_col_names)
    print('test_data column names : ', test_col_names)

    train_rows = train_rows[1:]
    test_rows = test_rows[1:]

    if check_exist_middle_new_line(train_rows) or check_exist_middle_new_line(test_rows):
        print('[-] There are some newlines in the middle of strings')
        return

    ret_train_rows = []
    ret_test_rows = []

    for row in train_rows:
        new_row = [ s[1:-1] for s in row.split(',')]
        ret_train_rows.append(new_row)
    
    for row in test_rows:
        new_row = [ s[1:-1] for s in row.split(',')]
        ret_test_rows.append(new_row)

    return (ret_train_rows, ret_test_rows)

def main():
    train_rows, test_rows = read_datas()

if __name__ == '__main__':
    main()
