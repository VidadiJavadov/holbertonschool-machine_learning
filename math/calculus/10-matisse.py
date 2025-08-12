#!/usr/bin/env python3
"""Derivative of polies"""


def poly_derivative(poly):
    """function that find that derivative"""
    if isinstance(poly[0], int):
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
    return None
