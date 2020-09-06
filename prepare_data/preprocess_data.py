import os
import re
import unicodedata

import tqdm

from prepare_data.load_data import LoadData
from utils.load_config import LoadConfig




class PreprocessText:
    def __init__(self, config, train_dataset, valid_dataset, test_dataset, store_as_file=False):

        self.config = config
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.store_as_file = store_as_file


    def _decode_string(self, dataset):
        lang_one = [str(lang_one.numpy().decode()) for lang_one, lang_two in dataset]
        lang_two = [str(lang_two.numpy().decode()) for lang_one, lang_two in dataset]
        return lang_one, lang_two



    def _tfds_to_txt(self, lang_one, lang_two, dataset_format):
        lang_one_path = os.path.join(self.config['preprocess_text_path'], f'{dataset_format}_{self.config["lang_one_file"]}')
        lang_two_path = os.path.join(self.config['preprocess_text_path'], f'{dataset_format}_{self.config["lang_two_file"]}')

        with open(lang_one_path, 'w', encoding='utf-8') as lang_one_file:
            for line in lang_one:
                lang_one_file.write(line+'\n')

        with open(lang_two_path, 'w', encoding='utf-8') as lang_two_file:
            for line in lang_two:
                lang_two_file.write(line+'\n')


    def _unicode_to_ascii(self, s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


    def _preprocess_sentence(self, w):
        w = self._unicode_to_ascii(w.lower().strip())
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)
        w = re.sub(r"[^a-zA-Z.!,¿]+", " ", w)
        w = w.strip()
        return w


    def _initial_preprocess_sentence(self, lang_one, lang_two):
        lang_one_result = []
        lang_two_result = []

        for sent in tqdm.tqdm(lang_one):
            lang_one_result.append(self._preprocess_sentence(sent))

        for sent in tqdm.tqdm(lang_two):
            lang_two_result.append(self._preprocess_sentence(sent))

        return lang_one_result, lang_two_result


    def clean_text(self):
        train_lang_1, train_lang_2 = self._decode_string(self.train_dataset)
        valid_lang_1, valid_lang_2 = self._decode_string(self.valid_dataset)
        test_lang_1, test_lang_2 = self._decode_string(self.test_dataset)

        print('\nProcessing Cleansing...')
        train_lang_1, train_lang_2 = self._initial_preprocess_sentence(train_lang_1, train_lang_2)
        valid_lang_1, valid_lang_2 = self._initial_preprocess_sentence(valid_lang_1, valid_lang_2)
        test_lang_1, test_lang_2 = self._initial_preprocess_sentence(test_lang_1, test_lang_2)

        if self.store_as_file:
            print(f'\nInitial Preprocessed Text Saved at {self.config["preprocess_text_path"]}')
            os.makedirs(self.config['preprocess_text_path'], exist_ok=True)
            self._tfds_to_txt(train_lang_1, train_lang_2, 'train')
            self._tfds_to_txt(valid_lang_1, valid_lang_2, 'valid')
            self._tfds_to_txt(test_lang_1, test_lang_2, 'test')

        train_data = (train_lang_1, train_lang_2)
        valid_data = (valid_lang_1, valid_lang_2)
        test_data = (test_lang_1, test_lang_2)

        return train_data, valid_data, test_data




if __name__ =='__main__':
    config_dict = LoadConfig('conf').load_config()
    dataset_name = config_dict['dataset_name']
    data_gen = LoadData(dataset_name=dataset_name)
    train_d, valid_d, test_d, infos = data_gen.get_data()
    tmp_processor = PreprocessText(config_dict, train_d, valid_d, test_d, True)
    a, b, c = tmp_processor.clean_text()
    print(a[0][0])
    print(a[1][0])

