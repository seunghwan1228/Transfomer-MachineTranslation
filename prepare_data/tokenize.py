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
        lang_one_tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus((lang_one for lang_one in lang_one_list),
                                                                                     target_vocab_size=self.config['vocab_size'])

        lang_two_tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus((lang_two for lang_two in lang_two_list),
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
                                               self.config['model_prefix']+'_lang_one',
                                               self.config['vocab_size'],
                                               self.config['character_coverage'],
                                               self.config['model_type'])

        lang_two_cmd = training_templte.format(lang_two_path,
                                               self.config['pad_id'],
                                               self.config['bos_id'],
                                               self.config['eos_id'],
                                               self.config['unk_id'],
                                               self.config['model_prefix']+'_lang_two',
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

        lang_one_tokenizer.Load(self.config['model_prefix'] + '_lang_one' + '.model')
        lang_two_tokenizer.Load(self.config['model_prefix'] + '_lang_two' + '.model')
        return lang_one_tokenizer, lang_two_tokenizer



    # TODO: REQUIRES TO BUILD BASIC TOKENIZER
    def _word_tokenizer(self, lang_one_list, lang_two_list):
        tokenizer_one = tf.keras.preprocessing.text.Tokenizer(num_words=config['vocab_size'],
                                                              oov_token='<UNK>')

        tokenizer_two = tf.keras.preprocessing.text.Tokenizer(num_words=config['vocab_size'],
                                                              oov_token='<UNK>')

        tokenizer_one.fit_on_texts(lang_one_list)
        tokenizer_two.fit_on_texts(lang_two_list)

        return tokenizer_one, tokenizer_two

    # TODO: REQQUIRES TO BUILD ONE METHOD TO TOKENIZE OTHER TOKENIZERS
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

        return lang_one_tokenized, lang_two_tokenized



    def _convert_to_ids(self, lang_one_list, lang_two_list, lang_one_tokenizer, lang_two_tokenizer):
        pass



    # TODO: REQUIRES TO ADD <SOS> & <EOS> TOKEN
    def _subword_add_special_token(self, tokenizer):
        sos_token = tokenizer.vocab_size
        eos_token = tokenizer.vocab_size + 1
        return sos_token, eos_token







if __name__ == '__main__':
    config = LoadConfig('conf').load_config()
    train_d, valid_d, test_d, infos = LoadData(config['dataset_name']).get_data()
    t, vd, ttd = PreprocessText(config, train_d, valid_d, test_d, True).clean_text()

    tokenizer = TokenizeData(config, t)

    sample_text = ['hellow this is the test sentence',
                   'this is the test sentence']

    # # Toknizer Test  -- Subword
    lang_one_tok_sub, lang_two_tok_sub = tokenizer._subword_tokenizer(t[0], t[1])
    lang_two_tok_sub.encode('hello, this is the test sentence, for the subword tokenizer')

    lang_two_tok_sub.encode(sample_text)

    lang_two_tok_sub.vocab_size

    # # Tokenizer Test -- Sentencepiece
    # tokenizer._sentencepiece_tokenizer_trainer()
    #
    # lang_one_tok_spm, lang_two_tok_spm = tokenizer._load_sentencepiece()
    # lang_two_tok_spm.EncodeAsIds('hello, this is the test sentence, for the sentence piece tokenizer')
    # lang_two_tok_spm.EncodeAsPieces('hello, this is the test sentence, for the sentence piece tokenizer')

    # Tokenizer TEst -- Word Tokenize
    # lang_one_tok_tk, lang_two_tok_tk = tokenizer._word_tokenize(t[0], t[1])
