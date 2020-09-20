import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def plot_attention_weights(attention_weights: dict,
                           attention_part:str, # encoder or decoder's attention dict key
                           sentence: str,
                           result: list,
                           data_creator):

    fig = plt.figure(figsize=(16, 8))
    inp_sentence_converted = data_creator.convert_to_ids([sentence], [], False)
    inp_sentence_converted = inp_sentence_converted[0]

    # ATTENTION stored in dictionary
    #{Decoder_Layer_1_0th_Attentention_Weight: (B, H, seq_len, seq_len)}
    attn = attention_weights[attention_part]
    attn = tf.squeeze(attn, axis=0) # (H, seq_len_q, seq_len_kv)
    # if selected layer is decoder_2,
    # seq_len_q is lang2
    # seq_len_kv is lang1

    for head in range(attn.shape[0]):
        ax = fig.add_subplot(2, 4, head+1)

        ax.matshow(attn[head][:-1, :], cmap='viridis') # remove EOS token

        fontdict = {'fontsize':10}
        ax.set_xtics(range(len(inp_sentence_converted)+2))
        ax.set_ytics(range(len(result)))
        ax.set_ylim(len(result)-1.5, -0.5)


        ax.set_xticklabels(['<sos>'] + [data_creator.convert_to_texts(i,
                                                                      data_creator.tokenizer.lang_one_tokenizer,
                                                                      data_creator.tokenizer.lang_one_sos,
                                                                      data_creator.tokenizer.lang_one_eos) for i in inp_sentence_converted] + ['<eos>'],
                           fontdict=fontdict,
                           rotation=90)


        ax.set_yticklabels([data_creator.convert_to_texts(i,
                                                          data_creator.tokenizer.lang_two_tokenizer,
                                                          data_creator.tokenizer.lang_two_sos,
                                                          data_creator.lang_two_eos) for i in result if i < data_creator.tokenizer.lang_two_vocab_size],
                           fontdict=fontdict)

        ax.set_xlabels('Head {}'.format(head))

    plt.tight_layout()
    plt.show()








