import gensim
import pandas as pd 
import numpy as np
import os
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
PATH_TO_FORMATTED_CSV = 'data/formatted.csv'
def load_embeddings():
    model = gensim.models.KeyedVectors.load_word2vec_format('embeddings/GoogleNews-vectors-negative300.bin', binary=True)
    return model 

def load_dataset():
    if os.path.exists(PATH_TO_FORMATTED_CSV):
        df = pd.read_csv(PATH_TO_FORMATTED_CSV)
        return _to_tensors(df)

    df = pd.read_csv('data/train.csv')

    df = df.drop(['id'], axis=1)

    df['comment_text'].fillna('unknown')
    df['toxic'].fillna(0)
    df['severe_toxic'].fillna(0)
    df['obscene'].fillna(0)
    df['threat'].fillna(0)
    df['insult'].fillna(0)
    df['identity_hate'].fillna(0)


    make_csv_one_hot_encoded(df=df)
    load_dataset()

def _format_toxic(row):

    rows = []
    if row['toxic'] == 1 :

        row_toxic = row.copy()
        row_toxic[1:7] = np.zeros(shape=(6,))
        row_toxic['toxic'] = 1
        rows.append(row_toxic)

    if row['severe_toxic'] == 1 :

        row_severe_toxic = row.copy()
        row_severe_toxic[1:7] = np.zeros(shape=(6,))
        row_severe_toxic['severe_toxic'] = 1
        rows.append(row_severe_toxic)
    if row['obscene'] == 1 :

        row_obscene = row.copy()
        row_obscene[1:7] = np.zeros(shape=(6,))
        row_obscene['obscene'] = 1
        rows.append(row_obscene)
    if row['threat'] == 1 :

        row_threat= row.copy()
        row_threat[1:7] = np.zeros(shape=(6,))
        row_threat['obscene'] = 1
        rows.append(row_threat)

    if row['insult'] == 1 :

        row_insult = row.copy()
        row_insult[1:7] = np.zeros(shape=(6,))
        row_insult['insult'] = 1
        rows.append(row_insult)

    if row['identity_hate'] == 1 :

        row_identity_hate = row.copy()
        row_identity_hate[1:7] = np.zeros(shape=(6,))
        row_identity_hate['identity_hate'] = 1
        rows.append(row_identity_hate)

    return rows


def make_csv_one_hot_encoded(df):

    formatted_df = df.copy()

    for index , row in df.iterrows():
        if index % 1000 == 0 :
            print(index)
        if (row[1:7].values != np.zeros(shape=(6,) , dtype= int)).any() :

            new_rows = _format_toxic(row)

            formatted_df = formatted_df.drop(df.index[index])
            formatted_df = formatted_df.append(new_rows)

    formatted_df.to_csv('data/formatted.csv')
    formatted_df = formatted_df.sample(frac=1).reset_index(drop=True)
    formatted_df.to_csv('data/formatted.csv')


def _to_tensors(df):
    comments = df['comment_text'].as_matrix()

    labels = df.iloc[:, 2:8].as_matrix()

    comments = tf.data.Dataset.from_tensor_slices(comments)

    labels = tf.data.Dataset.from_tensor_slices(labels)

    return comments, labels


def main():
    comments  , labels =  load_dataset()

    print(comments.output_shapes)
    print(labels.output_shapes)

    pass

if __name__ == "__main__":
    main()