import tensorflow as tf
import tqdm
import time

from models.masking import create_padding_mask, create_combined_mask
from utils.load_config import LoadConfig
from prepare_data.create_data import CreateData



# Config dict and model is for reference
config_dict = LoadConfig('conf').load_config()


# Load Data
dataset_name = config_dict['dataset_name']
data_creator = CreateData(config_path='conf')
train_datasets, valid_datasets, test_datasets = data_creator.create_all()


def evaluate(inp_sentence, model, data_creator, max_length):
    inp_sentence_converted = data_creator.tokenizer.convert_to_ids([inp_sentence], [], False)
    inp_sentence_converted = inp_sentence_converted[0]
    inp_sentence_converted = tf.constant(inp_sentence_converted)

    decoder_input = [data_creator.tokenizer.lang_two_sos]
    translate_result = tf.expand_dims(decoder_input, 0)

    for i in range(max_length):
        enc_padding = create_padding_mask(inp_sentence_converted)
        combined_mask = create_combined_mask(translate_result)

        prediction, encoder_attn, decoder_attn = model(encoder_input = inp_sentence_converted,
                                                     decoder_input = translate_result,
                                                     encoder_mask = enc_padding,
                                                     decoder_mask_one = combined_mask,
                                                     decoder_mask_two = enc_padding,
                                                     drop_n_heads = config_dict['drop_n_heads'],
                                                     training=False)

        prediction = prediction[:, -1:, :] # softmax result (B, seq_len, vocab_size)
        prediction_id = tf.argmax(prediction, axis=-1)
        prediction_id = tf.cast(prediction_id, tf.int32)

        if prediction_id == data_creator.tokenizer.lang_two_eos:
            return translate_result, encoder_attn, decoder_attn

        translate_result = tf.concat([translate_result, prediction_id], axis=-1)

    return translate_result, encoder_attn, decoder_attn


