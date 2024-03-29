#!/usr/bin/env python3
""" Attention """
import tensorflow as tf

SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """ Decode for machine translation """
    def __init__(self, vocab, embedding, units, batch):
        """ initialization """
        super(RNNDecoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units, recurrent_initializer="glorot_uniform",
            return_sequences=True, return_state=True)
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(self.units)

    def call(self, x, s_prev, hidden_states):
        """ Output word + Decoder hidden state """
        context_vector, _ = self.attention(s_prev, hidden_states)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, hidden = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.F(output)
        return x, hidden
