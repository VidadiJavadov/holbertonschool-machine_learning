#!/usr/bin/env python3
"""Adding matrices"""


def shape_of_mat(mat1):
    """calculate shape of a matrix"""
    shape = []
    while(isinstance(mat1, list)):
        shape.append(len(mat1))
        mat1 = mat1[0]
    return shape

def checking_shapes(mat1, mat2):
    """checking equality of two matrices"""
    if(shape_of_mat(mat1) == shape_of_mat(mat2)):
        return True
    return False



def add_matrices(mat1, mat2):
    """adding two matrices with each other"""
    if(checking_shapes(mat1,mat2)):
        res = []
        if(isinstance(mat1, list)):
            for i, j in zip(mat1, mat2):
                res.append(add_matrices(i, j))
            return res
        else:
            return mat1 + mat2
    return None
