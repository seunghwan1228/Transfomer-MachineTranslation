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