#!/usr/bin/env python3
"""Elementwise operations in matrices"""


def np_elementwise(mat1, mat2):
    """function elementwise"""
    sum = mat1 + mat2
    dif = mat1 - mat2
    mult = mat1 * mat2
    div = mat1 / mat2
    return sum, dif, mult, div
