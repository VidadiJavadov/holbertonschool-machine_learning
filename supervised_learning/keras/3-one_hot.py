#!/usr/bin/env python3
import tensorflow.keras as K
"""one hot funct"""


def one_hot(labels, classes=None):
    """Converts a label vector into a one-hot encoded matrix."""
    return K.utils.to_categorical(labels, num_classes=classes)
