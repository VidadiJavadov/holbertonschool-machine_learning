#!/usr/bin/env python3
"""Finding integral of poly"""


def poly_integral(poly, C=0):
    """function that finds the integral"""
    if (not isinstance(poly, list) or len(poly) == 0
            or not all(isinstance(c, (int, float)) for c in poly)):
        return None

    res = [C]
    for i in range(len(poly)):
        if poly[i]%(i+1) == 0:
            res.append(int(poly[i]/(i+1)))
        elif poly[i]%(i+1) != 0:
            res.append(float(poly[i]/(i+1)))
    return res
