#!/usr/bin/env python3
import tensorflow as tf

SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """RNN Decoder with attention for machine translation"""

    def __init__(self, vocab, embedding, units, batch):
        """Initialize decoder layers"""
        super(RNNDecoder, self).__init__()

        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab,
            output_dim=embedding
        )

        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=False,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """
        Perform one decoding step with attention
        """
        # x shape: (batch, 1)
        x = self.embedding(x)

        # Context vector from attention
        context, _ = self.attention(s_prev, hidden_states)

        # Concatenate context and embedding
        context = tf.expand_dims(context, 1)
        x = tf.concat([context, x], axis=-1)

        # GRU step
        _, s = self.gru(x, initial_state=s_prev)

        # Output prediction
        y = self.F(s)

        return y, s
