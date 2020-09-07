import numpy as np
import tensorflow as tf


def create_padding_mask(seq):
    # Mask 1 if the padding value
    # 1 == not look
    mask = tf.math.equal(seq, 0)
    mask = tf.cast(mask, tf.float32)
    return mask


def create_lookahed_mask(size):
    # Mask where should not look
    # 1 == not look
    mask = tf.linalg.band_part(tf.ones(shape = (size, size)), -1, 0)
    mask = 1 - mask
    return mask


def create_combined_mask(seq):
    # Combined padding and lookahead mask
    seq_len = tf.shape(seq)[1]
    padding_mask = create_padding_mask(seq)
    look_ahead_mask = create_lookahed_mask(seq_len)
    combined_mask = tf.math.maximum(padding_mask, look_ahead_mask)
    return combined_mask