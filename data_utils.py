from DataHandler import DataHandler


if __name__ == '__main__':
    data_handler = DataHandler(
        path_to_data_folder= 'data/'
    )
    data_handler.load_dataset()