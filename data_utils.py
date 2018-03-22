import tensorflow as tf 
import gensim
import pandas as pd 
import numpy as np

labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

def load_embeddings():
    model = gensim.models.KeyedVectors.load_word2vec_format('embeddings/GoogleNews-vectors-negative300.bin', binary=True)
    return model 

def load_data_set():

    df = pd.read_csv('data/train.csv')
    df.drop(['id'], axis=1)

    df['comment_text'].fillna('unknown')
    df['toxic'].fillna(0)
    df['severe_toxic'].fillna(0)
    df['obscene'].fillna(0)
    df['threat'].fillna(0)
    df['insult'].fillna(0)


    labels =   df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].values
    make_one_hot_encoded(df=df)



def make_one_hot_encoded(df):
    for index , row in df.iterrows():
        if row == np.zeros(shape=(5,0)):
            print('Index {} is clean '.format(index))
            return

def main():
    # My code here
    load_data_set()
    pass

if __name__ == "__main__":
    main()