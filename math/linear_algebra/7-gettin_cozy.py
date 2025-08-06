#!/usr/bin/env python3
"""concatenates arrays"""


def cat_matrices2D(mat1, mat2, axis=0):
    """function which returns cat_arrays"""
    if not mat1 or not mat2 or not isinstance(mat1[0], list) or not isinstance(mat2[0], list):
        return None

    if axis == 0:
        res = []
        if len(mat1[0]) != len(mat2[0]):
            return None
        for row in mat1:
            res.append(row)

        for row in mat2:
            res.append(row)
        return res

    if axis == 1:
        res = []
        if len(mat1) != len(mat2):
            return None
        for row1, row2 in zip(mat1, mat2):
            res.append(row1 + row2)
        return res
