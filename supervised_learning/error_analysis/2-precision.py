#!/usr/bin/env python3
"""precision"""
import numpy as np


def precision(confusion):
    """precision"""
    precision = []
    for i in range(confusion[0]):
        TP = confusion[i, i]
        FP = np.sum(confusion[:, i]) - TP
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0
        precision.append(prec)
    return np.array(precision)
