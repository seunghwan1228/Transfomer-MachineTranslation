import tensorflow as tf
import os

from utils.load_config import LoadConfig
from prepare_data.load_data import LoadData
from prepare_data.preprocess_data import PreprocessText
from prepare_data.tokenize import TokenizeData
from prepare_data.create_data import CreateData

from utils.print_config import PrintConfig




data_creator = CreateData(config_path='conf')
PrintConfig(data_creator.config).print_config()

train_datasets, valid_datasets, test_datasets = data_creator.create_all()


print('\nTrain Data\n')
for lang_1, lang_2 in train_datasets.take(1):
    # print(lang_1)
    print(lang_2)
    decoded = data_creator.tokenizer.convert_to_texts(lang_2[0].numpy(),
                                                      data_creator.tokenizer.lang_two_tokenizer,
                                                      data_creator.tokenizer.lang_two_sos,
                                                      data_creator.tokenizer.lang_two_eos)
    print(decoded)


print('\nValid Data\n')
for lang_1, lang_2 in valid_datasets.take(1):
    # print(lang_1)
    print(lang_2)
    decoded = data_creator.tokenizer.convert_to_texts(lang_2[0].numpy(),
                                                      data_creator.tokenizer.lang_two_tokenizer,
                                                      data_creator.tokenizer.lang_two_sos,
                                                      data_creator.tokenizer.lang_two_eos)
    print(decoded)


print('\nTest Data\n')
for lang_1, lang_2 in test_datasets.take(1):
    # print(lang_1)
    print(lang_2)
    decoded = data_creator.tokenizer.convert_to_texts(lang_2[0].numpy(),
                                                      data_creator.tokenizer.lang_two_tokenizer,
                                                      data_creator.tokenizer.lang_two_sos,
                                                      data_creator.tokenizer.lang_two_eos)
    print(decoded)