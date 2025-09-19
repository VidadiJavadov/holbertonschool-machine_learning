#!/usr/bin/env python3
"""Create a layer with using dropout technique"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob,training=True):
    """Dropout layer"""
    
    initializer = tf.keras.initializers.VarianceScaling(mode="fan_avg")

    dense = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer
    )(prev)

    dropout = tf.keras.layers.Dropout(rate=(1 - keep_prob))(dense, training=training)

    return dropout
