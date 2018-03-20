import tensorflow as tf 
import gensim
import pandas as pd 
import numpy as np 
labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
def load_embeddings():
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    return model 

def one_hot_encode(label):
    vector = np.zeros(shape=(6,))
    for i in range(len(labels)):
        if label == labels[i]:
            vector[i] = 1 
    
    return vector
