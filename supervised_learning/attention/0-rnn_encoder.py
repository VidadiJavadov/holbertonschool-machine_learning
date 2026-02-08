#!/usr/bin/python3
import tensorflow as tf
"""rnn encoder"""

class RNNEncoder(tf.keras.layers.Layer):
    """
    RNN Encoder for machine translation
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Initializes the encoder
        """
        super(RNNEncoder, self).__init__()

        self.batch = batch
        self.units = units

        # Embedding layer
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab,
            output_dim=embedding
        )

        # GRU layer
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

    def initialize_hidden_state(self):
        """
        Initializes the hidden state to zeros
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        Forward pass of the encoder
        """
        # x shape: (batch, input_seq_len)
        x = self.embedding(x)

        # outputs shape: (batch, input_seq_len, units)
        # hidden shape: (batch, units)
        outputs, hidden = self.gru(x, initial_state=initial)

        return outputs, hidden
