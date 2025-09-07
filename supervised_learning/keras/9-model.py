#!/usr/bin/env python3
"""save and load model"""

import tensorflow.keras as K


def save_model(network, filename):
    """Saves an entire model."""
    network.save(filename)
    return None


def load_model(filename):
    """Loads an entire model."""
    return K.models.load_model(filename)
