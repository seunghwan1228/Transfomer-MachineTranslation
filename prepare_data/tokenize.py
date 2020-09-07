import tensorflow as tf
import tensorflow_datasets as tfds
import sentencepiece as spm
import os

from utils.load_config import LoadConfig
from prepare_data.load_data import LoadData
from prepare_data.preprocess_data import PreprocessText


class TokenizeData:
    def __init__(self, config, train_data, add_start_end=True):
        self.config = config
        self.train_data = train_data
        self.add_start_end = add_start_end

    def _data_overview(self, dataset):
        sample_data = {}
        for lang_one, lang_two in dataset.take(1):
            sample_data['Language_One'] = lang_one
            sample_data['Language_Two'] = lang_two

        return sample_data

    def _subword_tokenizer(self,
                           lang_one_list: list,
                           lang_two_list: list):
        lang_one_tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (lang_one for lang_one in lang_one_list),
            target_vocab_size=self.config['vocab_size'])

        lang_two_tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (lang_two for lang_two in lang_two_list),
            target_vocab_size=self.config['vocab_size'])

        return lang_one_tokenizer, lang_two_tokenizer

    def _sentencepiece_tokenizer_trainer(self):
        print('To Perfome the Sentence Piece Please run Preprocessor_data to store .txt file !!')
        os.makedirs(config['spm_model_saver'], exist_ok=True)

        print(f'The Sentence Piece Model will save at {config["spm_model_saver"]}')
        training_templte = '--input={} \
        --pad_id={} \
        --bos_id={} \
        --eos_id={} \
        --unk_id={} \
        --model_prefix={} \
        --vocab_size={} \
        --character_coverage={} \
        --model_type={}'

        lang_one_path = os.path.join(self.config['preprocess_text_path'], f'train_{self.config["lang_one_file"]}')
        lang_two_path = os.path.join(self.config['preprocess_text_path'], f'train_{self.config["lang_two_file"]}')
        print(lang_one_path)
        lang_one_cmd = training_templte.format(lang_one_path,
                                               self.config['pad_id'],
                                               self.config['bos_id'],
                                               self.config['eos_id'],
                                               self.config['unk_id'],
                                               self.config['model_prefix'] + '_lang_one' + f'_{self.config["model_type"]}',
                                               self.config['vocab_size'],
                                               self.config['character_coverage'],
                                               self.config['model_type'])

        lang_two_cmd = training_templte.format(lang_two_path,
                                               self.config['pad_id'],
                                               self.config['bos_id'],
                                               self.config['eos_id'],
                                               self.config['unk_id'],
                                               self.config['model_prefix'] + '_lang_two' + f'_{self.config["model_type"]}',
                                               self.config['vocab_size'],
                                               self.config['character_coverage'],
                                               self.config['model_type'])
        print('\nLanguage One CMD:: \n')
        print(lang_one_cmd)
        print('\nLanguage Two CMD:: \n')
        print(lang_two_cmd)

        spm.SentencePieceTrainer.Train(lang_one_cmd)
        spm.SentencePieceTrainer.Train(lang_two_cmd)

    def _load_sentencepiece(self):
        lang_one_tokenizer = spm.SentencePieceProcessor()
        lang_two_tokenizer = spm.SentencePieceProcessor()

        lang_one_tokenizer.Load(self.config['model_prefix'] + f'_lang_one_{self.config["model_type"]}' + '.model')
        lang_two_tokenizer.Load(self.config['model_prefix'] + f'_lang_two_{self.config["model_type"]}' + '.model')
        return lang_one_tokenizer, lang_two_tokenizer

    def _word_tokenizer(self, lang_one_list, lang_two_list):
        tokenizer_one = tf.keras.preprocessing.text.Tokenizer(num_words=config['vocab_size'],
                                                              oov_token='<UNK>')

        tokenizer_two = tf.keras.preprocessing.text.Tokenizer(num_words=config['vocab_size'],
                                                              oov_token='<UNK>')

        tokenizer_one.fit_on_texts(lang_one_list)
        tokenizer_two.fit_on_texts(lang_two_list)

        return tokenizer_one, tokenizer_two

    # To create one function to convert text to ids
    def _subword_tokenizer_to_ids(self,
                                  lang_one_list,
                                  lang_two_list,
                                  sub_word_tokenizer_one,
                                  sub_word_tokenizer_two):

        lang_one_tokenized = []
        lang_two_tokenized = []

        lang_one_sos, lang_one_eos = self._subword_add_special_token(sub_word_tokenizer_one)
        lang_two_sos, lang_two_eos = self._subword_add_special_token(sub_word_tokenizer_two)

        for sentence in lang_one_list:
            if self.add_start_end:
                lang_one_encoded = [lang_one_sos] + sub_word_tokenizer_one.encode(sentence) + [lang_one_eos]
            else:
                lang_one_encoded = sub_word_tokenizer_one.encode(sentence)
            lang_one_tokenized.append(lang_one_encoded)

        for sentence in lang_two_list:
            if self.add_start_end:
                lang_two_encoded = [lang_two_sos] + sub_word_tokenizer_two.encode(sentence) + [lang_two_eos]
            else:
                lang_two_encoded = sub_word_tokenizer_two.encode(sentence)
            lang_two_tokenized.append(lang_two_encoded)

        self.lang_one_sos, self.lang_one_eos = lang_one_sos, lang_one_eos
        self.lang_two_sos, self.lang_two_eos = lang_two_sos, lang_two_eos

        return lang_one_tokenized, lang_two_tokenized


    def _word_tokenizer_to_ids(self,
                               lang_one_list,
                               lang_two_list,
                               word_tokenizer_one,
                               word_tokenizer_two):
        lang_one_tokenized = []
        lang_two_tokenized = []

        lang_one_sos, lang_one_eos = self._word_tokenizer_add_special_token(word_tokenizer_one)
        lang_two_sos, lang_two_eos = self._word_tokenizer_add_special_token(word_tokenizer_two)

        for sentence in lang_one_list:
            if self.add_start_end:
                lang_one_encoded = word_tokenizer_one.texts_to_sequences([sentence])
                lang_one_encoded = [lang_one_sos] + lang_one_encoded[0] + [lang_one_eos]
            else:
                lang_one_encoded = word_tokenizer_one.texts_to_sequences([sentence])
            lang_one_tokenized.append(lang_one_encoded)

        for sentence in lang_two_list:
            if self.add_start_end:
                lang_two_encoded = word_tokenizer_two.texts_to_sequences([sentence])
                lang_two_encoded = [lang_two_sos] + lang_two_encoded[0] + [lang_two_eos]
            else:
                lang_two_encoded = word_tokenizer_two.texts_to_sequences([sentence])
            lang_two_tokenized.append(lang_two_encoded)

        # word_tokenizer_one.index_word['0'] = '<SOS>'
        # word_tokenizer_one.index_word[f'{len(word_tokenizer_one.word_index) + 1}'] = '<EOS>'

        self.lang_one_sos, self.lang_one_eos = lang_one_sos, lang_one_eos
        self.lang_two_sos, self.lang_two_eos = lang_two_sos, lang_two_eos

        return lang_one_tokenized, lang_two_tokenized


    def _sentencepiece_tokenizer_to_ids(self, lang_one_list, lang_two_list, lang_one_tokenizer, lang_two_tokenizer):
        lang_one_tokenized = []
        lang_two_tokenized = []

        if self.add_start_end:
            self._spm_add_special_token(lang_one_tokenizer)
            self._spm_add_special_token(lang_two_tokenizer)

        for sentence in lang_one_list:
            lang_one_encoded = lang_one_tokenizer.EncodeAsIds(sentence)
            lang_one_tokenized.append(lang_one_encoded)

        for sentence in lang_two_list:
            lang_two_encoded = lang_two_tokenizer.EncodeAsIds(sentence)
            lang_two_tokenized.append(lang_two_encoded)

        self.lang_one_sos, self.lang_one_eos = self.config['bos_id'], self.config['eos_id']
        self.lang_two_sos, self.lang_two_eos = self.config['bos_id'], self.config['eos_id']

        lang_one_vocab_size = lang_one_tokenizer.vocab_size()
        lang_two_vocab_size = lang_two_tokenizer.vocab_size()
        return lang_one_tokenized, lang_two_tokenized

    def _convert_to_ids(self, lang_one_list, lang_two_list):
        '''
        Tokenizer Encode Wrapper
        '''
        print(f'Toknizer Type:: {self.config["tokenizer"]}')

        if self.config["tokenizer"] == 'subword':
            print('Create Tokenizer for Language One & Language Two\nSubword Tokenizer Requires some time to fit...')

        if self.config['tokenizer'] == 'subword':
            lang_one_tokenizer, lang_two_tokenizer = self._subword_tokenizer(lang_one_list, lang_two_list)
            lang_one_result, lang_two_result = self._subword_tokenizer_to_ids(lang_one_list, lang_two_list,
                                                                              lang_one_tokenizer, lang_two_tokenizer)

            lang_one_vocab_size = lang_one_tokenizer.vocab_size + 3
            lang_two_vocab_size = lang_two_tokenizer.vocab_size + 3

        elif self.config['tokenizer'] == 'sentencepiece':

            print(f'Training Sentence Piece... \n')
            self._sentencepiece_tokenizer_trainer()
            lang_one_tokenizer, lang_two_tokenizer = self._load_sentencepiece()
            lang_one_result, lang_two_result = self._sentencepiece_tokenizer_to_ids(lang_one_list, lang_two_list,
                                                                                    lang_one_tokenizer,
                                                                                    lang_two_tokenizer)

            lang_one_vocab_size = self.config['vocab_size']
            lang_two_vocab_size = self.config['vocab_size']

        elif self.config['tokenizer'] == 'word':
            lang_one_tokenizer, lang_two_tokenizer = self._word_tokenizer(lang_one_list, lang_two_list)
            lang_one_result, lang_two_result = self._word_tokenizer_to_ids(lang_one_list, lang_two_list,
                                                                           lang_one_tokenizer, lang_two_tokenizer)

            lang_one_vocab_size = len(lang_one_tokenizer.word_index) + 3
            lang_two_vocab_size = len(lang_two_tokenizer.word_index) + 3

        else:
            raise ValueError(
                f'The Config -- tokenizer should be in <subword>, <sentencepiece>, or <word> NOT {self.config["tokenizer"]}')

        self.lang_one_tokenizer = lang_one_tokenizer
        self.lang_two_tokenizer = lang_two_tokenizer
        self.lang_one_vocab_size = lang_one_vocab_size
        self.lang_two_vocab_size = lang_two_vocab_size

        return lang_one_result, lang_two_result

    def _convert_to_texts(self, language_id, tokenizer, sos_token_num, eos_token_num):
        '''
        Tokenizer Decode Wrapper
        '''
        # Trim token
        if language_id[0] == sos_token_num:
            language_id = language_id[1:]

        if language_id[-1] == eos_token_num:
            language_id = language_id[:-1]

        if self.config['tokenizer'] == 'subword':
            decoded_language_id = tokenizer.decode(language_id)

        elif self.config['tokenizer'] == 'sentencepiece':
            decoded_language_id = tokenizer.DecodeIds(language_id)

        elif self.config['tokenizer'] == 'word':
            decoded_language_id = tokenizer.sequences_to_texts([language_id])[0]

        return decoded_language_id


    def _subword_add_special_token(self, tokenizer):
        sos_token = tokenizer.vocab_size + 1
        eos_token = tokenizer.vocab_size + 2
        return sos_token, eos_token

    def _spm_add_special_token(self, tokenizer):
        return tokenizer.SetEncodeExtraOptions('bos:eos')

    def _word_tokenizer_add_special_token(self, tokenizer):
        sos_token = len(tokenizer.index_word) + 1
        eos_token = len(tokenizer.index_word) + 2
        return sos_token, eos_token





if __name__ == '__main__':
    config = LoadConfig('conf').load_config()
    train_d, valid_d, test_d, infos = LoadData(config['dataset_name']).get_data()
    t, vd, ttd = PreprocessText(config, train_d, valid_d, test_d, True).clean_text()

    # Define Tokenizer
    tokenizer = TokenizeData(config, t)


    # Tokenizer Test
    encode_one, encode_two = tokenizer._convert_to_ids(t[0], t[1])

    decode_one = tokenizer._convert_to_texts(encode_one[0], tokenizer.lang_one_tokenizer, tokenizer.lang_one_sos, tokenizer.lang_one_eos)
    decode_two = tokenizer._convert_to_texts(encode_two[0], tokenizer.lang_two_tokenizer, tokenizer.lang_two_sos, tokenizer.lang_two_eos)

    # Checker
    print(f'Tokenizer Method::{config["tokenizer"]}\n')
    print('Original Sentence\n')
    print(f'Pt\tEn\n')
    print(f'{t[0][0]}\n{t[1][0]}\n')
    print('*-- Encoded to Ids --*')
    print(f'{encode_one[0]}\n{encode_two[0]}\n')
    print('*-- Decode to Text --*')
    print(f'{decode_one}\n{decode_two}')