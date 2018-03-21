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
    df.drop(['id'],axis=1)
    toxic = df['toxic']
    comments = df['comment_text']
    severe_toxic = df['severe_toxic']
    obscene = df['obscene']
    threat = df['threat']
    insult = df['insult']
    identity_hate = df['identity_hate']
    np.concatenate([toxic, com])