#!/usr/bin/env python3
"""Builds a transition layer as described in DenseNet (DenseNet-C)."""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """Builds a transition layer with compression."""
    he_normal = K.initializers.HeNormal(seed=0)

    # Batch Normalization + ReLU
    bn = K.layers.BatchNormalization(axis=3)(X)
    act = K.layers.Activation('relu')(bn)

    # 1x1 Convolution for compression
    nb_filters = int(nb_filters * compression)
    conv = K.layers.Conv2D(
        nb_filters, (1, 1),
        padding='same', kernel_initializer=he_normal
    )(act)

    # Average Pooling (2x2, stride 2)
    X_out = K.layers.AveragePooling2D((2, 2), strides=(2, 2))(conv)

    return X_out, nb_filters
