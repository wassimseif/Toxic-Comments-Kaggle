import tensorflow as tf 
import gensim
import pandas as pd 
import numpy as np

labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

def load_embeddings():
    model = gensim.models.KeyedVectors.load_word2vec_format('embeddings/GoogleNews-vectors-negative300.bin', binary=True)
    return model 

def one_hot_encode(label):
    vector = np.zeros(shape=(6,))
    for i in range(len(labels)):
        if label == labels[i]:
            vector[i] = 1 
    
    return vector

def load_data_set():

    df = pd.read_csv('data/train.csv')
    df.drop(['id'], axis=1)

    df['comment_text'].fillna('unknown')
    df['toxic'].fillna(0)
    df['severe_toxic'].fillna(0)
    df['obscene'].fillna(0)
    df['threat'].fillna(0)
    df['insult'].fillna(0)



    comments = df['comment_text']
    toxic = df['toxic']
    severe_toxic = df['severe_toxic']
    obscene = df['obscene']
    threat = df['threat']
    insult = df['insult']
    identity_hate = df['identity_hate']
    labels = np.vstack([toxic, severe_toxic, obscene,threat,insult,identity_hate], axis = 0)
    print(labels.shape)


def main():
    # My code here
    load_data_set()
    pass

if __name__ == "__main__":
    main()