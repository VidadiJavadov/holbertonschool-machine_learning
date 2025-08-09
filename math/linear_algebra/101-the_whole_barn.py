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
        for i, j in zip(mat1, mat2):
            if(isinstance(i, list)):
                for k, m in zip(i, j):
                    if(isinstance(k, list)):
                        for l, t in zip(k, m):
                            if(isinstance(l, list)):
                                for h, u in zip(l,t):
                                    res.append(h+u)
                    else:
                        res.append(k+m)
            else:
                res.append(i+j)
        return res
