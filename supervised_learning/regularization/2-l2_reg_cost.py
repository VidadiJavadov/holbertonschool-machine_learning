#!/usr/bin/env python3
"""L2 reg cost"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """function that computes L2 reg"""

    l2_loss = tf.add_n(model.losses)
    total_cost = [cost + l for l in l2_loss]
    return tf.convert_to_tensor(total_cost)
