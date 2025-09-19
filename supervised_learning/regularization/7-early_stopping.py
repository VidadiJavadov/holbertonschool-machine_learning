#!/usr/bin/env python3
"""Early stopping"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Early stopping"""
    if cost < opt_cost - threshold:
        opt_cost = cost
        count = 0
    
    else:
        count+=1

    stop = count >= patience
    return stop, count

print(early_stopping(1.0, 1.9, 0.5, 15, 5))
print(early_stopping(1.1, 1.5, 0.5, 15, 2))
print(early_stopping(1.0, 1.5, 0.5, 15, 8))
print(early_stopping(1.0, 1.5, 0.5, 15, 14))