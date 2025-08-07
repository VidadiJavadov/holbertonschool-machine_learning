#!/usr/bin/env python3
"""matrix multiplication"""


def mat_mul(mat1, mat2):
    """The function that returns multiple of two matrices"""
    if len(mat1[0]) != len(mat2):
        return None

    res = [[0 for i in range(len(mat2[0]))] for j in range(len(mat1))]

    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                res[i][j] += mat1[i][k] * mat2[k][j]
    return res
