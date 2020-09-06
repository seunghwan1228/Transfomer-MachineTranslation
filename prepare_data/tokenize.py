import tensorflow as tf
import tensorflow_datasets as tfds
import sentencepiece as spm
import os

from utils.load_config import LoadConfig
from prepare_data.load_data import LoadData
from prepare_data.preprocess_data import PreprocessText



class TokenizeData:
    def __init__(self, config, train_data):
        self.config = config
        self.train_data = train_data


    def _data_overview(self, dataset):
        sample_data = {}
        for lang_one, lang_two in dataset.take(1):
            sample_data['Language_One'] = lang_one
            sample_data['Language_Two'] = lang_two

        return sample_data



    def _subword_tokenizer(self, lang_one_list, lang_two_list):
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

    # TODO: REQQUIRES TO BUILD ONE METHOD TO TOKENIZE OTHER TOKENIZERS

    # TODO: REQUIRES TO ADD <SOS> & <EOS> TOKEN




if __name__ == '__main__':
    config = LoadConfig('conf').load_config()
    train_d, valid_d, test_d, infos = LoadData(config['dataset_name']).get_data()
    t, vd, ttd = PreprocessText(config, train_d, valid_d, test_d, True).clean_text()

    tokenizer = TokenizeData(config, t)

    # Toknizer Test  -- Subword
    lang_one_tok_sub, lang_two_tok_sub = tokenizer._subword_tokenizer(t[0], t[1])
    lang_two_tok_sub.encode('hello, this is the test sentence, for the subword tokenizer')

    # Tokenizer Test -- Sentencepiece
    tokenizer._sentencepiece_tokenizer_trainer()

    lang_one_tok_spm, lang_two_tok_spm = tokenizer._load_sentencepiece()
    lang_two_tok_spm.EncodeAsIds('hello, this is the test sentence, for the sentence piece tokenizer')
    lang_two_tok_spm.EncodeAsPieces('hello, this is the test sentence, for the sentence piece tokenizer')
