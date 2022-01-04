#!/usr/bin/env python3.10

"""
Create string by repeating literan n times and return number of chars.
"""

import cboost

@cboost.make_cpp
def wiecej(n: int) -> int:
    return 100 * n

@cboost.make_cpp
def test_map():
    liczby: list = [1,3,5,7,9]

    for n in liczby:
        print(f'{n = }')

    for n in map(wiecej, liczby):
        print(f'{n = }')

if __name__ == '__main__':
    import sys
    try:
        ...
    except IndexError:
        print(f'usage: {sys.argv[0]}')
        sys.exit(1)

    test_map()
