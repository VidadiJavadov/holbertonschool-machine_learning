#!/usr/bin/env python3
"""Create layer with dropout"""
import tensorflow as tf

def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """Layer with dropout"""
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')
    weights = initializer(shape=(prev.shape[1], n))
    bias = tf.zeros([n])

    layer_output = tf.matmul(prev, weights) + bias

    activated_output = activation(layer_output)

    if training:
        activated_output = tf.nn.dropout(activated_output, rate=1 - keep_prob)
    
    return activated_output
