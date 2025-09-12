#!/usr/bin/env python3
"""Momentum optimizer in TensorFlow"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """create optimizer"""
    op = tf.keras.optimizers.Adam(alpha, beta1)
    return op
