#!/usr/bin/env python3
"""self attention"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Self-attention layer for machine translation"""

    def __init__(self, units):
        """Initialize attention layers"""
        super(SelfAttention, self).__init__()

        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        Compute context vector and attention weights
        """
        # s_prev shape: (batch, units)
        # hidden_states shape: (batch, input_seq_len, units)

        # Expand s_prev to match time dimension
        s_prev = tf.expand_dims(s_prev, 1)

        # Alignment scores
        score = self.V(
            tf.nn.tanh(self.W(s_prev) + self.U(hidden_states))
        )

        # Attention weights
        weights = tf.nn.softmax(score, axis=1)

        # Context vector
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
