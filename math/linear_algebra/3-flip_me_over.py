#!/usr/bin/env python3
"""Trnspose of matrix"""

def matrix_transpose(matrix):
    """function part"""
    rows = len(matrix[0])
    columns = len(matrix)
    result = []
    for i in range(rows):
        new_row = []
        for j in range(columns):
            new_row.append(matrix[j][i])
        result.append(new_row)

    return result
