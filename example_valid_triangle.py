#!/usr/bin/env python3.10

"""
Check if given side lengths make a valid triangle.

https://en.wikipedia.org/wiki/Triangle_inequality
"""

import cboost

@cboost.make_cpp
def is_valid_triangle(a: int, b: int, c: int) -> bool:
    conds: 'auto' = [(a+b)>c, (b+c)>a, (a+c)>b]

    return all(conds)

if __name__ == '__main__':
    import sys
    try:
        a, b, c = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    except IndexError:
        print(f'usage: {sys.argv[0]} A B C')
        sys.exit(1)

    print(f'{is_valid_triangle(a, b, c) = }')

