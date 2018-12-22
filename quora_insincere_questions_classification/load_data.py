import os
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from collections import Counter
from keras_preprocessing.sequence import pad_sequences

base_dir = os.getcwd()
vocab_fname = 'vocab.txt'
train_fname = 'train.csv'
test_fname = 'test.csv'


def build_vocab(sentences, unk_limit):
    words = []

    for sent in sentences:
        words.extend(sent)

    word_count = Counter(words)
    words = [pair[0] for pair in word_count.most_common()[:unk_limit]]

    vocb = dict()
    vocb['<PAD>'] = 0
    vocb['<UNK>'] = 1

    for i, word in enumerate(words):
        vocb[word] = i + 2

    # save vocb
    with open(os.path.join(base_dir, vocab_fname), 'w', encoding='utf-8') as f:
        for word, idx in vocb.items():
            f.write('{}\t{}\n'.format(word, idx))

    return vocb


# read csv files
def load_train_data(unk_limit):
    file_path = os.path.join(base_dir, train_fname)
    train_data = pd.read_csv(file_path, quotechar='"')
    train_data = train_data.sample(frac=1)

    train_x = []
    train_y = []
    avg_len = 0
    max_len = 0

    # extract features and labels
    for idx, row in enumerate(train_data.iterrows()):
        # only when developing
        if idx > 100:
            break
        sent, label = row[1][1:]
        sent = sent.lower()

        words = word_tokenize(sent)
        avg_len += len(words)
        max_len = max(max_len, len(words))

        train_x.append(words)
        train_y.append(label)

    train_y = np.asarray(train_y)
    train_y = train_y.astype(np.float)

    avg_len /= len(train_x)
    print('average of sentence length : ', avg_len)
    print('max length sentence : ', max_len)

    return train_x, train_y

def word2idx(sentences, vocb):
    sent_as_idx = []

    for sent in sentences:
        sent_as_idx.append([vocb.get(word, vocb['<UNK>']) for word in sent])

    return sent_as_idx


def load_data(unk_limit):
    (train_x, train_y) = load_train_data(unk_limit)

    assert len(train_x) == len(train_y)

    vocb = build_vocab(train_x, unk_limit)

    train_x = word2idx(train_x, vocb)

    return train_x, train_y
