#!/usr/bin/env python3
import tensorflow as tf
"Simplest autoencoder"


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates an autoencoder.

    input_dims: int, dimension of the input
    hidden_layers: list of ints, number of nodes in each encoder hidden layer
    latent_dims: int, dimension of the latent space

    Returns: encoder, decoder, auto
    """

    # -------- Encoder --------
    encoder_input = tf.keras.Input(shape=(input_dims,))
    x = encoder_input

    for units in hidden_layers:
        x = tf.keras.layers.Dense(units, activation='relu')(x)

    latent = tf.keras.layers.Dense(latent_dims, activation='relu')(x)

    encoder = tf.keras.Model(encoder_input, latent)

    # -------- Decoder --------
    decoder_input = tf.keras.Input(shape=(latent_dims,))
    x = decoder_input

    for units in reversed(hidden_layers):
        x = tf.keras.layers.Dense(units, activation='relu')(x)

    decoder_output = tf.keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = tf.keras.Model(decoder_input, decoder_output)

    # -------- Autoencoder --------
    auto_output = decoder(encoder(encoder_input))
    auto = tf.keras.Model(encoder_input, auto_output)

    auto.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )

    return encoder, decoder, auto
