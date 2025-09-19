#!/usr/bin/env python3
"""Create a layer with using dropout technique"""
import keras as tf


def dropout_create_layer(prev, n, activation, keep_prob,training=True):
    """Dropout layer"""
    
    initializer = tf.keras.initializers.VarianceScaling(mode="fan_avg")

    dense = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer
    )(prev)

    if training:
        dropout = tf.layers.Dropout(rate=(1 - keep_prob))(dense, training=True)
        return dropout
    else:
        return dense
