#!/usr/bin/env python3

"""
Check if given angles make right triangle.

https://en.wikipedia.org/wiki/Triangle#Right_triangles
"""

import cboost

@cboost.make_cpp
def is_right_triangle(alpha: int, beta: int) -> bool:
    gamma: int = 180 - alpha - beta
    right: auto = 90
    conds: 'auto' = [alpha == right, beta == right, gamma == right]

    return any(conds)

if __name__ == '__main__':
    import sys
    try:
        alpha, beta = int(sys.argv[1]), int(sys.argv[2])
    except IndexError:
        print(f'usage: {sys.argv[0]} ALPHA BETA')
        sys.exit(1)

    print(f'{is_right_triangle(alpha, beta) = }')

