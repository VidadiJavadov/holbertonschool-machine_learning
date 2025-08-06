#!/usr/bin/env python3
"""adding matrices"""


def matrix_shape(matrix):
    """function for shape"""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        if len(matrix) == 0:
            break
        matrix = matrix[0]
    return shape


def add_matrices2D(mat1, mat2):
    """function for adding 2d matrices with same shape"""
    res = []
    if matrix_shape(mat1) == matrix_shape(mat2):
        for i,j in zip(mat1, mat2):
            new_row = []
            for n,k in zip(i,j):
                new_row.append(n+k)
            res.append(new_row)
        return res
    else:
        return None
