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
    Softmax_k {(Q @ K^t) / sqrt(k_dim)} @ V    <- Attn_weight @ V
    '''
    def __init__(self, **kwargs):
        super(ScaledDotAttention, self).__init__(**kwargs)

    def call(self, q, k, v, mask):
        # Q: (B, seq_len_q, dim) | (B, H, seq_len_q, depth)
        # K: (B, seq_len_k, dim) | (B, H, seq_len_k, depth)
        # V: (B, seq_len_v, dim) | (B, H, seq_len_v, depth)  Where seq_len_k == seq_len_v, define as seq_len_kv
        # Mask: (B, length) := {Masked == 1}
        # Mask Shape should be the same
        # 3-D: (B, 1, seq_len_q)
        # 4-D: (B, 1, 1, seq_len_q)

        qk_mul = tf.matmul(q, k, transpose_b=True)    # (B, seq_len_q, seq_len_kv) | (B, H, seq_len_q, seq_len_kv)
        k_dim = tf.cast(tf.shape(k)[-1], tf.float32)

        att_logit = qk_mul / k_dim                    # (B, seq_len_q, seq_len_kv) | (B, H, seq_len_q, seq_len_kv)

        if mask is not None:
            att_logit += (mask * -1e9)                # Add requires the same shape - Broad cast H, seq_len_q

        att_weight = tf.nn.softmax(att_logit, axis=-1) # (B, seq_len_q, seq_len_kv) | (B, H, seq_len_q, seq_len_kv)
        context_vector = tf.matmul(att_weight, v)      # (B, seq_len_q, seq_len_kv) @ (B, seq_len_kv, dim) == (B, seq_len_q, dim)   <- 3-D Tensor
                                                       # (B, H, seq_len_q, seq_len_kv) @ (B, H, seq_len_kv, depth) == (B, H, seq_len_q, depth)  <- 4-D Tensor
        return context_vector, att_weight




class MultiHeadAttention(tf.keras.layers.Layer):
    '''
    Call Values
    - Q, K, V, MASK, DROP_N_HEADS, TRAINING

    RETURN VALUES
    - CONTEXT, ATTENTION_WEIGHT
    '''
    def __init__(self, num_heads, model_dim, concat_query, debug=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.concat_query = concat_query

        assert self.model_dim % self.num_heads == 0, "The Model Dimension should be divided by num_heads"

        self.depth = self.model_dim // self.num_heads

        self.WQ = tf.keras.layers.Dense(units=model_dim)
        self.WK = tf.keras.layers.Dense(units=model_dim)
        self.WV = tf.keras.layers.Dense(units=model_dim)

        self.scaled_attention = ScaledDotAttention()

        self.linear = tf.keras.layers.Dense(units=model_dim)

        self.debug = debug
    def split_head(self, x):
        '''
        Input: (B, seq_len, dim)
        '''
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        x = tf.reshape(x, shape=(batch_size, seq_len, self.num_heads, self.depth)) # (b, seq_len, H, depth)
        x = tf.transpose(x, perm=[0, 2, 1, 3])                                     # (b, H, seq_len, depth)
        return x

    def concat_head(self, x):
        '''
        Input: (B, H, seq_len, dim)
        '''
        batch_size = tf.shape(x)[0]
        x = tf.transpose(x, perm=[0, 2, 1, 3])                                     # (b, seq_len, H, depth)
        x = tf.reshape(x, shape=(batch_size, -1, self.model_dim))                  # (b, seq_len, model_dim)  Where model_dim = (depth * H)
        return x

    def drop_head(self, x, drop_n_heads, training):
        """
        * -- This Method is applied from -- *
        Scheduled DropHead: A Regularization Method for Transformer Models
        http://arxiv.org/abs/2004.13342

        x: input tensor, its shape (B, H, seq_len, depth)
        drop_n_heads: drop_head_number, integer.. May change rate in future. This implementation uses just integer
        """
        batch_size = tf.shape(x)[0]
        assert (self.num_heads - drop_n_heads) != 0, 'To Apply Drop_head, "Num_heads" > "Drop_n_heads", not "Num_heads == "Drop_n_heads"'

        if training != True:
            return x

        if drop_n_heads == 0:
            return x

        remain_heads = tf.ones(shape=(self.num_heads - drop_n_heads), dtype=tf.float32)
        drop_heads = tf.zeros(shape=(drop_n_heads), dtype=tf.float32)
        head_drop_mask = tf.concat((remain_heads, drop_heads), axis=-1)

        tf_array = tf.TensorArray(dtype=tf.float32, size=batch_size)
        for i in range(batch_size):
            shuffled_head_drop_mask = tf.random.shuffle(head_drop_mask)
            tf_array = tf_array.write(i, shuffled_head_drop_mask)

        tf_array = tf_array.stack()                                         # (B, H)
        tf_array = tf_array[..., tf.newaxis, tf.newaxis]                    # (b, H, 1, 1)

        drop_result = tf.math.multiply(x, tf_array)

        lambda_value = (self.num_heads - drop_n_heads) / self.num_heads
        lambda_value = tf.cast(lambda_value, tf.float32)

        return drop_result / lambda_value

    def call(self, q, k, v, mask, drop_n_heads, training):
        Q = self.WQ(q)                                              # b, seq_len_q,  dim
        K = self.WK(k)                                              # b, seq_len_vk, dim
        V = self.WV(v)                                              # b, seq_len_vk, dim

        # Where depth = H // model_dim
        q_split = self.split_head(Q)                                # b, H, seq_len_q,  depth
        k_split = self.split_head(K)                                # b, H, seq_len_vk, depth
        v_split = self.split_head(V)                                # b, H, seq_len_vk, depth

        context_vector, attention_weight = self.scaled_attention(q_split, k_split, v_split, mask)  # b, H, seq_len_q, depth

        # Apply Head Drop
        context_vector = self.drop_head(context_vector, drop_n_heads, training)  # b, H, seq_len_q, depth
        if self.debug:
            print(f'Context Vector After Drop Head Applied..\n', context_vector)

        context_vector = self.concat_head(context_vector)   # b, seq_len, model_dim


        if self.concat_query:
            context_vector = tf.concat((context_vector, q), axis=-1) # b, seq_len, 2*model_dim

        if self.debug:
            print(f'Context Vector Before Linear Applied..\n', context_vector)

        linear_out = self.linear(context_vector)

        return linear_out, attention_weight


class FeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, feed_forward_dim, model_dim, dropout_rate, **kwargs):
        super(FeedForwardNetwork, self).__init__(**kwargs)
        self.feed_forward_dim = feed_forward_dim
        self.model_dim = model_dim
        self.dropout_rate = dropout_rate

        self.d1 = tf.keras.layers.Dense(feed_forward_dim)
        self.d1_act = tf.keras.layers.Activation('relu')
        self.dr1 = tf.keras.layers.Dropout(dropout_rate)

        self.d2 = tf.keras.layers.Dense(model_dim)
        self.d2_act = tf.keras.layers.Activation('relu')
        self.dr2 = tf.keras.layers.Dropout(dropout_rate)

        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-9)


    def call(self, inputs, training):
        input_res = inputs                  # (B, seq_len, model_dim)
        x = self.d1(inputs)                 # (B, seq_len ,feedforward_dim)
        x = self.d1_act(x)                  # (B, seq_len ,feedforward_dim)
        x = self.dr1(x)                     # (B, seq_len ,feedforward_dim)
        x = self.d2(x)                      # (B, seq_len ,model_dim)
        x = self.d2_act(x)                  # (B, seq_len ,model_dim)
        x = self.dr2(x)                     # (B, seq_len ,model_dim)
        res_con = input_res + x             # (B, seq_len ,model_dim)
        output_ = self.layer_norm(res_con)  # (B, seq_len ,model_dim)
        return output_



class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, model_dim,  dropout_rate, mha_concat_query, debug = False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.num_heads= num_heads
        self.model_dim = model_dim
        self.dropout_rate = dropout_rate
        self.concat_query = mha_concat_query
        self.debug = debug

        self.mha = MultiHeadAttention(num_heads, model_dim, concat_query = self.concat_query, debug=self.debug)






if __name__ == '__main__':
    from models.masking import create_padding_mask
    print('This is Testing Layer Test')
    # Create dummy sequence
    tmp_seq_1 = tf.random.uniform(shape=(2, 8), minval=0, maxval=100, dtype=tf.int32)
    tmp_pad_1 = tf.zeros(shape=(2, 2), dtype=tf.int32)

    tmp_seq_2 = tf.random.uniform(shape=(2, 4), minval=0, maxval=100, dtype=tf.int32)
    tmp_pad_2 = tf.zeros(shape=(2, 6), dtype=tf.int32)

    input_seq_1 = tf.concat((tmp_seq_1, tmp_pad_1), axis=-1)
    input_seq_2 = tf.concat((tmp_seq_2, tmp_pad_2), axis=-1)


    # 1) Where Q = K = V
    input_seq = tf.concat((input_seq_1, input_seq_2), axis=0)
    print(input_seq)

    padding_mask = create_padding_mask(input_seq)
    print(padding_mask)

    # Batch: 4
    # seq_len: 10
    # dim: 4
    emb_out = tf.keras.layers.Embedding(101, 4)(input_seq)
    print(emb_out.shape)

    mha_no_concat = MultiHeadAttention(2, 4, False, debug=True)
    context_vector, attn_weights = mha_no_concat(q= emb_out, k= emb_out, v= emb_out, mask = padding_mask, drop_n_heads=1, training=True)

    print('Final Context Vector', context_vector)
    print('Final Attention Weights', attn_weights)