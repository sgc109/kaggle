import pandas as pd
import numpy as np
from keras import layers, models
from sklearn.model_selection import train_test_split
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from sklearn import metrics

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print('train shape : ', train_df.shape)
print('test shape : ', test_df.shape)

train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2018)

embed_size = 300
max_features = 50000
max_len = 100

train_x = train_df['question_text'].fillna('__na__').values
val_x = val_df['question_text'].fillna('__na__').values
test_x = test_df['question_text'].fillna('__na__').values

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_x))

train_x = tokenizer.texts_to_sequences(train_x)
val_x = tokenizer.texts_to_sequences(val_x)
test_x = tokenizer.texts_to_sequences(test_x)

train_x = pad_sequences(train_x, maxlen=max_len)
val_x = pad_sequences(val_x, maxlen=max_len)
test_x = pad_sequences(test_x, maxlen=max_len)

train_y = train_df['target'].values
val_y = val_df['target'].values

model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(max_len,)))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=512, epochs=2, validation_data=(val_x, val_y))

pred_test = model.predict(test_x, batch_size=512)

for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_test > thresh).astype(int))))

# pred_test = (pred_test > 0.5).astype(int)
# out_df =
