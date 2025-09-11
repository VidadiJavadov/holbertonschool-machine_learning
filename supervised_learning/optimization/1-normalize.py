#!/usr/bin/env python3
"""normalize"""
import numpy as np


def normalize(X, m, s):
    """normalize"""
    X_norm = (X - m)/s
    return X_norm
