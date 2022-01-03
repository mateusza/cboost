#!/usr/bin/env python3.10

"""

"""

import cboost

@cboost.make_cpp
def test_abs():
    x: float = 100.0
    for i in range(10):
        x *= -0.9
        print(abs(x))

if __name__ == '__main__':
    import sys
    try:
        ...
    except IndexError:
        print(f'usage: {sys.argv[0]}')
        sys.exit(1)

    test_abs()

