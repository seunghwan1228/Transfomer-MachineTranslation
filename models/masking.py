import numpy as np
import tensorflow as tf


def create_padding_mask(seq):
    # Mask 1 if the padding value
    # 1 == not look
    mask = tf.math.equal(seq, 0)
    mask = tf.cast(mask, tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :] # (B, H, seq_len, dim)


def create_lookahed_mask(size):
    # Mask where should not look
    # 1 == not look
    mask = tf.linalg.band_part(tf.ones(shape = (size, size)), -1, 0)
    mask = 1 - mask
    return mask  # (seq_len, seq_len)


def create_combined_mask(seq):
    # Combined padding and lookahead mask
    seq_len = tf.shape(seq)[1]
    padding_mask = create_padding_mask(seq)
    look_ahead_mask = create_lookahed_mask(seq_len)
    combined_mask = tf.math.maximum(padding_mask, look_ahead_mask)
    return combined_mask


def create_mel_padding_mask(seq):
    '''
    This Function is for Audio Format input: Mel-Spectrogram to pad its shape
    '''
    # seq shape: b, seq_len, mel_channel

    mask = tf.abs(seq)
    mask = tf.reduce_sum(mask, axis=-1)
    mask = tf.math.equal(mask, 0)
    mask = tf.cast(mask, tf.float32)
    return mask


if __name__ == '__main__':
    tmp_input_seq = tf.random.uniform(shape=(2, 8))
    tmp_pad_seq = tf.zeros(shape=(2, 2))
    input_seq = tf.concat((tmp_input_seq, tmp_pad_seq), axis=1)

    print('Input Tensor')
    print(input_seq[0])

    print('\nPadding Mask -- from input seq')
    padding_mask = create_padding_mask(input_seq)
    print(padding_mask)

    print('\nLook Ahead Mask -- from input seq')
    lookahead_mask = create_lookahed_mask(size=input_seq.shape[-1])
    print(lookahead_mask)

    print('Combined Mask -- from input seq')
    combined_mask = create_combined_mask(input_seq)
    print(combined_mask)
