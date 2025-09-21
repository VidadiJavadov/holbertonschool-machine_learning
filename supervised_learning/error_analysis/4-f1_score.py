#!/usr/bin/env python3
"""f1 score"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """f1 score"""
    recall = sensitivity(confusion)
    prec = precision(confusion)

    f1_score = 2 * prec * recall / (prec + recall)
    return f1_score
