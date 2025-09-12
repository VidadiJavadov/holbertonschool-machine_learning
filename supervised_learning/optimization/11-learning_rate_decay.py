#!/usr/bin/env python3
"""Learning rate decay"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """learning rate decay"""
    step = np.floor(global_step/decay_step)
    new_alpha = alpha / (1+ decay_rate * step)
    return new_alpha
