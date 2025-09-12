#!/usr/bin/env python3
"""learning rate decay with tensorflow"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """learning rate decay with tensorflow"""

    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
