#!/usr/bin/env python3
"""test model"""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Tests a neural network and returns loss and accuracy."""
    return network.evaluate(x=data,
                            y=labels,
                            verbose=verbose)
