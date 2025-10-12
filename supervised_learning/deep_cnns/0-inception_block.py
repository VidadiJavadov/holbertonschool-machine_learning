#!/usr/bin/env python3
"""Defines an Inception block from 'Going Deeper with Convolutions' (2014)."""

from tensorflow import keras as K


def inception_block(A_prev, filters):
    """Builds an inception block and returns its concatenated output."""
    F1, F3R, F3, F5R, F5, FPP = filters

    conv1 = K.layers.Conv2D(
        filters=F1, kernel_size=(1, 1), padding='same', activation='relu'
    )(A_prev)

    conv3_reduce = K.layers.Conv2D(
        filters=F3R, kernel_size=(1, 1), padding='same', activation='relu'
    )(A_prev)
    conv3 = K.layers.Conv2D(
        filters=F3, kernel_size=(3, 3), padding='same', activation='relu'
    )(conv3_reduce)

    conv5_reduce = K.layers.Conv2D(
        filters=F5R, kernel_size=(1, 1), padding='same', activation='relu'
    )(A_prev)
    conv5 = K.layers.Conv2D(
        filters=F5, kernel_size=(5, 5), padding='same', activation='relu'
    )(conv5_reduce)

    max_pool = K.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(1, 1), padding='same'
    )(A_prev)
    max_pool_conv = K.layers.Conv2D(
        filters=FPP, kernel_size=(1, 1), padding='same', activation='relu'
    )(max_pool)

    output = K.layers.concatenate(
        [conv1, conv3, conv5, max_pool_conv], axis=-1
    )

    return output
