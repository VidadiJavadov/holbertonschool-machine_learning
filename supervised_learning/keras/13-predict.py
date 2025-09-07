#!/usr/bin/env python3
"""predict with model"""

import tensorflow.keras as K

def predict(network, data, verbose=False):
    """Makes a prediction using a neural network."""
    return network.predict(x=data, verbose=verbose)
