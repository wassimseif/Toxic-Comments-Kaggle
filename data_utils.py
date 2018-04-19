from DataHandler import DataHandler
from EmbeddingHandler import EmbeddingHandler


if __name__ == '__main__':

    data_handler = DataHandler(
        path_to_data_folder= 'data/'
    )
    data_handler.load_dataset()

    embedding_handler = EmbeddingHandler()