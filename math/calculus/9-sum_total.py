#!/usr/bin/env python3
"""Sum of i squared"""


def summation_i_squared(n):
    """summation i squared"""
    if not isinstance(n, int):
        return None
    return int(n*(n+1)*(2*n+1)/6)
