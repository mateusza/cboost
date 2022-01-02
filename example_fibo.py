#!/usr/bin/env python3.10

import cboost

@cboost.make_cpp
def fibo(n: int) -> int:
    if n in (0, 1):
        return n
    return fibo(n-2) + fibo(n-1)

if __name__ == '__main__':
    import sys
    try:
        n = int(sys.argv[1])
    except IndexError:
        print(f'usage: {sys.argv[0]} N')
        sys.exit(1)

    print(f'{fibo(n) = }')

