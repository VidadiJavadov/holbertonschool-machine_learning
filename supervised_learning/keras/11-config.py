#!/usr/bin/env python3
"""save and load model configuration"""

import tensorflow.keras as K


def save_config(network, filename):
    """Saves a model’s configuration in JSON format."""
    config = network.to_json()
    with open(filename, "w") as f:
        f.write(config)
    return None


def load_config(filename):
    """Loads a model with a specific configuration from JSON."""
    with open(filename, "r") as f:
        config = f.read()
    return K.models.model_from_json(config)
