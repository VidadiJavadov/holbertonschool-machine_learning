#!/usr/bin/env python3
"""Creating a layer with l2 reg"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Creating a layer with l2 reg"""
    l2_reg = tf.keras.regularizers.l2(lambtha*2)

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=l2_reg
    )

    return layer(prev)
