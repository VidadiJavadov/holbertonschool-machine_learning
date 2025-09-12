#!/usr/bin/env python3
"""RMSProp tensorflow"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """RMSProb tensorflow"""
    optimizer = tf.keras.optimizers.RMSprop(alpha, beta2, epsilon)
    return optimizer
