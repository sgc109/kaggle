import os
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from quora_insincere_questions_classification.load_data import load_data

base_dir = os.getcwd()
embedding_dir = os.path.join(base_dir, 'embeddings')
glove_dir = os.path.join(embedding_dir, 'glove.840B.300d')
glove_file_name = os.path.join(glove_dir, 'glove.840B.300d.txt')
unk_limit = 10000
emb_dim = 300
lr_rate = 0.0001
dev_rate = 0.1

train_x, train_y = load_data(unk_limit=unk_limit)

assert len(train_x) == len(train_y)

# load glove
with open(glove_file_name, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        if idx > 10:
            break
        line = line.rstrip()
        values = line.split()
        word = values[0]
        coef = values[1:]

