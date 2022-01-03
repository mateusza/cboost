#!/usr/bin/env python3.10

"""
Calculate number of primes less then given value (π(x))

https://en.wikipedia.org/wiki/Prime-counting_function
"""

import cboost

@cboost.make_cpp
def is_prime(n: int) -> bool:
    if n in (0, 1):
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    for d in range(3, n, 2):
        if n % d == 0:
            return False
        if d * d > n:
            break
    return True

@cboost.make_cpp
def how_many_primes(limit: int) -> int:
    how_many: int = 0
    for i in range(2, limit):
        if is_prime(i):
            how_many += 1
    return how_many

if __name__ == '__main__':
    import sys
    try:
        n = int(sys.argv[1])
    except IndexError:
        print(f'usage: {sys.argv[0]} N')
        sys.exit(1)

    print(f'{how_many_primes(n) = }')

