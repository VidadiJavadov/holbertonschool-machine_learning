#!/usr/bin/env python3
"""L2 reg cost"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """function that computes L2 reg"""

    l2_loss = tf.add_n(model.losses)
    return tf.stack(cost + l2_loss)
