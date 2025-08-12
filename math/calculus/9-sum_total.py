#!/usr/bin/env python3
def summation_i_squared(n):
    """summation i squared"""
    if not isinstance(n, int) and n < 1:
        return None
    return int(n*(n+1)*(2*n+1)/6)
