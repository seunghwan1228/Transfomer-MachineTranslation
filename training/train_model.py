import tensorflow as tf
from models.transformer import TransformerModel
from models.masking import create_padding_mask, create_combined_mask

from utils.load_config import LoadConfig


# Config dict and model is for reference
config_dict = LoadConfig('conf').load_config()


model = TransformerModel(encoder_vocab_size=config_dict['vocab_size'],
                         decoder_vocab_size=config_dict['vocab_size'],
                         encoder_max_pos=config_dict['max_pos_length'],
                         decoder_max_pos=config_dict['max_pos_length'],
                         num_heads=config_dict['num_heads'],
                         model_dim=config_dict['model_dim'],
                         feed_forward_dim=config_dict['feed_forward_dim'],
                         dropout_rate=config_dict['dropout_rate'],
                         mha_concat_query=config_dict['mha_concat_query'],
                         n_layers=config_dict['n_layers'],
                         debug=config_dict['debug'])



@tf.function
def model_train_step(model, encoder_input_seq, decoder_target_seq):
    decoder_input = decoder_target_seq[:, :-1]  # <sos> 1, 2, 3, 4, 5
    decoder_target = decoder_target_seq[:, 1:]  # 1, 2, 3, 4, <eos>

    encoder_padding = create_padding_mask(encoder_input_seq)
    decoder_padding_one = create_combined_mask(decoder_input)

    with tf.gradients() as tape:
        pass
