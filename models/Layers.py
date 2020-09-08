# This Document includes
# 1) Scaled Dot Attention
#    input: q, k, v
#    output: context_vector, attention_weight

# 2) Multi-Head Attention - includes Drop head
# 4) EncoderBlock - Based on Multi Head Attention
# 5) Encoder
# 6) DecoderBlock - Baed on Multi Head Attention
# 7) Decoder

import tensorflow as tf
import numpy as np


class ScaledDotAttention(tf.keras.layers.Layer):
    '''
    Softmax_k {(Q @ K^t) / sqrt(k_dim)} @ V
    '''
    def __init__(self, **kwargs):
        super(ScaledDotAttention, self).__init__(**kwargs)

    def call(self, q, k, v, mask):
        # Q: (B, seq_len_q, dim)
        # K: (B, seq_len_k, dim)
        # V: (B, seq_len_v, dim)   Where seq_len_k == seq_len_v, define as seq_len_kv
        # Mask: (B, length) := {Masked == 1}
        qk_mul = tf.matmul(q, k, transpose_b=True)    # (B, seq_len_q, seq_len_kv)
        k_dim = tf.cast(tf.shape(k)[-1], tf.float32)

        att_logit = qk_mul / k_dim  # (B, seq_len_q, seq_len_kv)

        if mask is not None:
            att_logit += (mask * -1e9)

        att_weight = tf.nn.softmax(att_logit, axis=-1) # (B, seq_len_q, seq_len_kv)
        context_vector = tf.matmul(v, att_weight)      # (B, seq_len_q, seq_len_kv) @ (B, seq_len_kv, dim) == (B, seq_len_q, dim)

        return context_vector, att_weight