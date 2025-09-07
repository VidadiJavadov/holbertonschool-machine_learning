#!/usr/bin/env python3
"""
Builds a neural network with Keras
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """build model"""
    model = K.Sequential()

    # First layer
    model.add(K.layers.Dense(
        units=layers[0],
        activation=activations[0],
        kernel_regularizer=K.regularizers.l2(lambtha),
        input_dim=nx
    ))

    # Add remaining layers with dropout (except output layer)
    for i in range(1, len(layers)):
        model.add(K.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        ))
        if i != len(layers) - 1 and keep_prob < 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
