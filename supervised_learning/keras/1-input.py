#!/usr/bin/env python3
"""
Builds a neural network with Keras (Functional API)
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a Keras model"""
    inputs = K.Input(shape=(nx,))
    x = inputs

    for i in range(len(layers)):
        x = K.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        )(x)
        if i != len(layers) - 1 and keep_prob < 1:
            x = K.layers.Dropout(1 - keep_prob)(x)

    return K.Model(inputs=inputs, outputs=x)
