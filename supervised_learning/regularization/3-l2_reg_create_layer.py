#!/usr/bin/env python3
"""Creating a layer with l2 reg"""
import keras as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Creating a layer with l2 reg"""
    l2_reg = tf.regularizers.l2(lambtha)

    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=l2_reg
    )

    return layer(prev)
