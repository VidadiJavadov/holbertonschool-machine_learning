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


def add_arrays(arr1, arr2):
    """function for adding matrices with same shape"""
    result = []
    if matrix_shape(arr1) == matrix_shape(arr2):
        for i, j in zip(arr1, arr2):
            result.append(i+j)
        return result
    else:
        return None
