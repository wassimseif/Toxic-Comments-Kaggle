import gensim
from gensim.models import Word2Vec

class EmbeddingHandler:
    def __init__(self):
        pass

    def read_pretrained_embeddings(self,embedding_file ):
        google_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    def train_embeddings(self,sentences):
        pass
