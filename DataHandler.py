import os
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.contrib.keras.api.keras.preprocessing.text import Tokenizer
from tensorflow.contrib.keras.api.keras.preprocessing import sequence


class DataHandler:

    vocab_size = 10000
    max_sentence_len = 200
    def __init__(self, path_to_data_folder):
        self.path_to_data_folder = path_to_data_folder

        if not self._check_if_formatted_csv_exsits():
            self._format_train_csv()

    def _check_if_formatted_csv_exsits(self):
        return os.path.exists(os.path.join(self.path_to_data_folder, 'formatted.csv'))

    def _format_train_csv(self):
        print('Formatting Train CSV')

        df = pd.read_csv(os.path.join(self.path_to_data_folder, 'train.csv'))

        df = df.drop(['id'], axis=1)

        df['comment_text'].fillna('unknown')
        df['toxic'].fillna(0)
        df['severe_toxic'].fillna(0)
        df['obscene'].fillna(0)
        df['threat'].fillna(0)
        df['insult'].fillna(0)
        df['identity_hate'].fillna(0)
        self.make_csv_one_hot_encoded(df)

    def make_csv_one_hot_encoded(self, df):

        formatted_df = df.copy()

        for index, row in df.iterrows():
            if index % 1000 == 0:
                print(index)
            if (row[1:7].values != np.zeros(shape=(6,), dtype=int)).any():
                new_rows = self._format_toxic(row)

                formatted_df = formatted_df.drop(df.index[index])
                formatted_df = formatted_df.append(new_rows)

        formatted_df.to_csv('data/formatted.csv')
        formatted_df = formatted_df.sample(frac=1).reset_index(drop=True)
        formatted_df.to_csv('data/formatted.csv')

    def _format_toxic(self, row):

        rows = []
        if row['toxic'] == 1:
            row_toxic = row.copy()
            row_toxic[1:7] = np.zeros(shape=(6,))
            row_toxic['toxic'] = 1
            rows.append(row_toxic)

        if row['severe_toxic'] == 1:
            row_severe_toxic = row.copy()
            row_severe_toxic[1:7] = np.zeros(shape=(6,))
            row_severe_toxic['severe_toxic'] = 1
            rows.append(row_severe_toxic)
        if row['obscene'] == 1:
            row_obscene = row.copy()
            row_obscene[1:7] = np.zeros(shape=(6,))
            row_obscene['obscene'] = 1
            rows.append(row_obscene)
        if row['threat'] == 1:
            row_threat = row.copy()
            row_threat[1:7] = np.zeros(shape=(6,))
            row_threat['obscene'] = 1
            rows.append(row_threat)

        if row['insult'] == 1:
            row_insult = row.copy()
            row_insult[1:7] = np.zeros(shape=(6,))
            row_insult['insult'] = 1
            rows.append(row_insult)

        if row['identity_hate'] == 1:
            row_identity_hate = row.copy()
            row_identity_hate[1:7] = np.zeros(shape=(6,))
            row_identity_hate['identity_hate'] = 1
            rows.append(row_identity_hate)

        return rows

    def load_dataset(self):
        if not self._check_if_formatted_csv_exsits():
            print('Formatted CSV NOT AVAILABLE')
            raise Exception
            return
        df = pd.read_csv(os.path.join(self.path_to_data_folder, 'formatted.csv'))

        comments = df['comment_text']

        self.tokenize(comments)

        comments = tf.data.Dataset.from_tensor_slices(comments)
        comments = comments.batch(3)

        labels = df.iloc[:, 2:8].as_matrix()
        labels = tf.data.Dataset.from_tensor_slices(labels)
        labels = labels.batch(3)

        return comments, labels

    def tokenize(self, comments):
        print('Comments shape is {}'.format(comments.shape))

        token = Tokenizer(
            num_words = self.vocab_size
        )
        token.fit_on_texts(comments)

        tokenized_comments = token.texts_to_sequences(comments)

        tokenized_comments = sequence.pad_sequences(
            sequences= tokenized_comments,
            maxlen= self.max_sentence_len,
            padding = 'post',
            value = 0
        )
        








