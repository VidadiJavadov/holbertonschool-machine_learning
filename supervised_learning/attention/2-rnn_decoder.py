#!/usr/bin/env python3
"""
RNNDecoder for Machine Translation
"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    Decoder class that inherits from tensorflow.keras.layers.Layer
    to decode for machine translation
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor
        """
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """
        Performs the decoding step
        """
        # Get context vector and attention weights
        context_vector, _ = self.attention(s_prev, hidden_states)

        # Embed the input word x: (batch, 1) -> (batch, 1, embedding)
        x = self.embedding(x)

        # Expand context vector: (batch, units) -> (batch, 1, units)
        context_expanded = tf.expand_dims(context_vector, 1)

        # Concatenate context vector with x: (batch, 1, units + embedding)
        x = tf.concat([context_expanded, x], axis=-1)

        # Pass through GRU
        output, s = self.gru(x)

        # Reshape output: (batch, 1, units) -> (batch, units)
        output = tf.reshape(output, (-1, output.shape[2]))

        # Pass through Dense layer to get output logits
        y = self.F(output)

        return y, s
