#!/usr/bin/env python3
"""
Multi Head Attention
"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    MultiHeadAttention class that inherits from tensorflow.keras.layers.Layer
    to perform multi head attention.
    """

    def __init__(self, dm, h):
        """
        Class constructor
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (h, depth).
        Transpose result to shape (batch_size, h, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Performs the multi head attention.
        """
        batch_size = tf.shape(Q)[0]

        # Linear layers
        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)

        # Split heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Scaled Dot Product Attention
        output, weights = sdp_attention(q, k, v, mask)

        # Concatenate heads
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.dm))

        # Final linear layer
        output = self.linear(concat_attention)

        return output, weights
