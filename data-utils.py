import tensorflow as tf 
import gensim
class Word2Vec:
    def load_embeddings(self):
        model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        return model

    @staticmethod
    def