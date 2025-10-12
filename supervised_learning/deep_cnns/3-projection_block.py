#!/usr/bin/env python3
"""Projection block for ResNet (He et al., 2015)."""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """Builds a projection block."""
    F11, F3, F12 = filters
    he_normal = K.initializers.HeNormal(seed=0)

    # Main path
    X = K.layers.Conv2D(
        F11, (1, 1), strides=(s, s),
        padding='valid', kernel_initializer=he_normal
    )(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(
        F3, (3, 3), padding='same',
        kernel_initializer=he_normal
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(
        F12, (1, 1), padding='valid',
        kernel_initializer=he_normal
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # Shortcut path
    shortcut = K.layers.Conv2D(
        F12, (1, 1), strides=(s, s),
        padding='valid', kernel_initializer=he_normal
    )(A_prev)
    shortcut = K.layers.BatchNormalization(axis=3)(shortcut)

    # Add shortcut and main path
    X = K.layers.Add()([X, shortcut])
    X = K.layers.Activation('relu')(X)

    return X
