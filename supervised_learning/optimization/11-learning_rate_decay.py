#!/usr/bin/env python3
"""Learning rate decay"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """learning rate decay"""
    alpha = alpha * decay_rate**(global_step/decay_step)
    return alpha
