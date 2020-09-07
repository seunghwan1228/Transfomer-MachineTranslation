import tensorflow as tf
import os

from utils.load_config import LoadConfig
from prepare_data.load_data import LoadData
from prepare_data.preprocess_data import PreprocessText
from prepare_data.tokenize import TokenizeData




class CreateData:
    def __init__(self, config_path,  store_as_file = False, add_start_end=True):
        self.config = LoadConfig(config_path).load_config()
        self.store_as_file = store_as_file
        self.add_start_end = add_start_end


        self.train_data, self.valid_data, self.test_data, self.data_info = LoadData(self.config['dataset_name']).get_data()

        if self.config['tokenizer'] == 'sentencepiece':
            self.store_as_file = True
        self.train_data, self.valid_data, self.test_data = PreprocessText(self.config,
                                                                          self.train_data,
                                                                          self.valid_data,
                                                                          self.test_data,
                                                                          store_as_file=self.store_as_file).clean_text()

        self.tokenizer = TokenizeData(self.config, self.train_data, add_start_end=self.add_start_end)

    def _tokenize_data(self, dataset, is_train):
        encoded_lang_one, encoded_lang_two = self.tokenizer._convert_to_ids(dataset[0], dataset[1], is_train=is_train)
        return encoded_lang_one, encoded_lang_two


    def _create_tf_dataset(self, dataset, is_train):
        encoded_lang_one, encoded_lang_two = self._tokenize_data(dataset, is_train)

        language_one = tf.ragged.constant(encoded_lang_one, dtype=tf.int32).to_tensor()
        language_one_data = tf.data.Dataset.from_tensor_slices(language_one)

        language_two = tf.ragged.constant(encoded_lang_two, dtype=tf.int32).to_tensor()
        language_two_data = tf.data.Dataset.from_tensor_slices(language_two)

        datasets = tf.data.Dataset.zip((language_one_data, language_two_data))
        datasets = datasets.shuffle(buffer_size=int(tf.shape(language_one)[0]))
        datasets = datasets.batch(self.config['batch_size'])
        datasets = datasets.prefetch(tf.data.experimental.AUTOTUNE)
        return datasets


    def create_all(self):
        train_datasets = self._create_tf_dataset(self.train_data, is_train=True)
        valid_datasets = self._create_tf_dataset(self.valid_data, is_train=False)
        test_datasets = self._create_tf_dataset(self.test_data, is_train=False)

        return train_datasets, valid_datasets, test_datasets











if __name__ == '__main__':
    data_creator = CreateData(config_path='conf')
    # lang_one, lang_two = data_creator.tokenize_data(data_creator.train_data)

    train_datasets, valid_datasets, test_datasets = data_creator.create_all()

    # TODO: REQUIRES TO COMPARE THE TOKENIZED ONE
    # data_creator.tokenizer.lang_one_tokenizer.word_index