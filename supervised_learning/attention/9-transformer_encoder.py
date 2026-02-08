#!/usr/bin/env python3
"""
Transformer Encoder
"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """
    Encoder class that inherits from tensorflow.keras.layers.Layer
    to create the encoder for a transformer.
    """

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Class constructor
        """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Performs the encoder pass.

        Args:
            x: Tensor of shape (batch, input_seq_len) containing input indices.
            training: Boolean, whether the model is training.
            mask: The mask to be applied for multi head attention.
        """
        seq_len = tf.shape(x)[1]

        # Adding embedding and rescaling by sqrt(dm)
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        # Add positional encoding
        # Slice positional encoding to match the sequence length
        x += self.positional_encoding[:seq_len]

        # Apply dropout
        x = self.dropout(x, training=training)

        # Pass through the N encoder blocks
        for i in range(self.N):
            x = self.blocks[i](x, training, mask)

        return x
