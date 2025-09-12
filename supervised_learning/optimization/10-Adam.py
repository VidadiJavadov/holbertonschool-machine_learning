#!/usr/bin/env python3
"""Adam optimizer using tensorflow"""
import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """Adam optimizer with tensorflow"""
    optimizer = tf.keras.optimizers.Adam(alpha, beta1, beta2, epsilon)
    return optimizer
