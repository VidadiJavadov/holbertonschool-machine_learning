#!/usr/bin/env python3
"""sensitivity"""
import numpy as np


def sensitivity(confusion):
    """sensitivity"""
    sensitivity = []
    for i in range(len(confusion)):
        TP = confusion[i, i]
        FN = np.sum(confusion[i, :]) - TP
        sens = TP / (TP + FN)
        sensitivity.append(sens)
    return np.array(sensitivity)
