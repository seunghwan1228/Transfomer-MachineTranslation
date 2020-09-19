import tensorflow as tf
from models.layers import Encoder, Decoder, DecoderProjection

class TransformerModel(tf.keras.models.Model):
    def __init__(self,
                 encoder_vocab_size,
                 decoder_vocab_size,
                 encoder_max_pos,
                 decoder_max_pos,
                 num_heads,
                 model_dim,
                 feed_forward_dim,
                 dropout_rate,
                 mha_concat_query,
                 n_layers,
                 debug=False,
                 **kwargs):
        super(TransformerModel, self).__init__(**kwargs)

        self.encoder = Encoder(vocab_size=encoder_vocab_size,
                               max_pos_length=encoder_max_pos,
                               num_heads=num_heads,
                               model_dim=model_dim,
                               feed_forward_dim=feed_forward_dim,
                               dropout_rate=dropout_rate,
                               mha_concat_query=mha_concat_query,
                               n_layers=n_layers,
                               debug=debug)

        self.decoder = Decoder(vocab_size=decoder_vocab_size,
                               max_pos_length=decoder_max_pos,
                               num_heads=num_heads,
                               model_dim=model_dim,
                               feed_forward_dim=feed_forward_dim,
                               dropout_rate=dropout_rate,
                               mha_concat_query=mha_concat_query,
                               n_layers=n_layers,
                               debug=debug)

        self.vocab_proj = DecoderProjection(decoder_vocab_size=decoder_vocab_size)


    def call(self, encoder_input, decoder_input, encoder_mask, decoder_mask_one, decoder_mask_two, drop_n_heads, training):
        encoder_output, encoder_attn_dict = self.encoder(encoder_input,
                                                         encoder_mask,
                                                         drop_n_heads,
                                                         training=training)
        decoder_output, decoder_attn_dict = self.decoder(decoder_input,
                                                         encoder_output,
                                                         decoder_mask_one,
                                                         decoder_mask_two,
                                                         drop_n_heads,
                                                         training=training)
        vocab_proj = self.vocab_proj(decoder_output)
        return vocab_proj, encoder_attn_dict, decoder_attn_dict



if __name__ == '__main__':
    from models.masking import create_padding_mask, create_combined_mask

    print('This is Testing Layer Test')

    def create_input_data():
        # Create dummy sequence
        tmp_seq_1 = tf.random.uniform(shape=(2, 8), minval=0, maxval=100, dtype=tf.int32)
        tmp_pad_1 = tf.zeros(shape=(2, 2), dtype=tf.int32)

        tmp_seq_2 = tf.random.uniform(shape=(2, 4), minval=0, maxval=100, dtype=tf.int32)
        tmp_pad_2 = tf.zeros(shape=(2, 6), dtype=tf.int32)

        input_seq_1 = tf.concat((tmp_seq_1, tmp_pad_1), axis=-1)
        input_seq_2 = tf.concat((tmp_seq_2, tmp_pad_2), axis=-1)

        # 1) Where Q = K = V
        input_seq = tf.concat((input_seq_1, input_seq_2), axis=0)
        return input_seq

    encoder_input = create_input_data()
    decoder_input = create_input_data()

    print('Encoder Input', encoder_input)
    print('Decoder Input', decoder_input)

    tmp_model = TransformerModel(101, 101, 101, 101, 2, 4, 4, 0.2, False, 2, True)


    encoder_mask = create_padding_mask(encoder_input)
    print('Encoder Mask', encoder_mask)

    combined_mask = create_combined_mask(decoder_input)
    print('Decoder Mask One', combined_mask)


    tmp_model_out, enc_attns, dec_attns = tmp_model(encoder_input,
                                                    decoder_input,
                                                    encoder_mask,
                                                    combined_mask,
                                                    encoder_mask,
                                                    1,
                                                    True)

    print('Model Output - SoftMaxed', tmp_model_out)

    print('Model Encoder Attention', enc_attns)
    print('Model Decoder Attention', dec_attns)
