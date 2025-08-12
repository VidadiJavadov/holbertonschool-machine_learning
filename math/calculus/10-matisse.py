#!/usr/bin/env python3
"""Derivative of polies"""


def poly_derivative(poly):
    """function that find that derivative"""
    if (not isinstance(poly, list) or len(poly) == 0
            or not all(isinstance(c, (float, int)) for c in poly)):
        return None

    res = []
    for i in range(1, len(poly)):
        res.append(i * poly[i])

    res2 = []
    for i in res:
        if i == 0:
            res2.append(i)

    if res == res2:
        return [0]
    return res
