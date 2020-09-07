import tensorflow as tf
import numpy as np



# Positional Encoding
# https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
# PE(pos, 2i)   = sin(pos / 10000 ^ (2i / d_model))
# PE(pos, 2i+1) = con(pos / 10000 ^ (2i / d_model))

def get_angles(pos, i, model_dim):
    angle_rates = 1 / np.power(10000, (2*(i//2)) / np.float32(model_dim))
    return pos * angle_rates


def positional_encoding(pos, model_dim):
    angle_rads = get_angles(np.arange(pos)[:, np.newaxis],          # pos, 1
                            np.arange(model_dim)[np.newaxis, :],    # 1, pos
                            model_dim)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return pos_encoding



class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def get_angles(self, pos, i, d_model): # pos: (seq_length, 1)  i: (1, d_model)
        angles = 1 / np.power(10000., (2 * (i // 2)) / np.float32(d_model))
        return pos * angles # (seq_length, d_model)

    def call(self, inputs):
        seq_length = inputs.shape.as_list()[-2]
        d_model = inputs.shape.as_list()[-1]

        angles = self.get_angles(np.arange(seq_length)[:, np.newaxis],
                                 np.arange(d_model)[np.newaxis, :],
                                 d_model)

        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])

        return tf.cast(angles[np.newaxis, ...], tf.float32) # (B, seq_length, d_model)





if __name__ == '__main__':
    import matplotlib.pyplot as plt
    pe = PositionalEncoding()
    pe_result = pe(tf.random.uniform(shape=(2, 5, 10)))
    plt.matshow(pe_result.numpy().squeeze())
    plt.show()