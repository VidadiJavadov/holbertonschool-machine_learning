#!/usr/bin/env python3
"""Builds a dense block as described in DenseNet (DenseNet-B)."""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Builds a dense block with bottleneck layers."""
    he_normal = K.initializers.HeNormal(seed=0)

    for i in range(layers):
        # Bottleneck layer: 1x1 conv
        bn1 = K.layers.BatchNormalization(axis=3)(X)
        act1 = K.layers.Activation('relu')(bn1)
        conv1 = K.layers.Conv2D(
            4 * growth_rate, (1, 1),
            padding='same', kernel_initializer=he_normal
        )(act1)

        # 3x3 conv
        bn2 = K.layers.BatchNormalization(axis=3)(conv1)
        act2 = K.layers.Activation('relu')(bn2)
        conv2 = K.layers.Conv2D(
            growth_rate, (3, 3),
            padding='same', kernel_initializer=he_normal
        )(act2)

        # Concatenate input and output (skip connection)
        X = K.layers.Concatenate(axis=3)([X, conv2])
        nb_filters += growth_rate

    return X, nb_filters
