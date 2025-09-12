#!/usr/bin/env python3
"""Batch normalization with tensorflow"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """batch norm with tensorflow"""

    dense = tf.keras.layers.Dense(
        n,
        kernel_initializer=tf.keras.initializers.VarianceScaling(mode='fan_avg'),
        use_bias=False
    )(prev)

    bn = tf.keras.layers.BatchNormalization(
        axis=-1,
        momentum=0.99,
        epsilon=1e-7,
        center=True,
        scale=True   
    )(dense)

    output = activation(bn)
    return output
