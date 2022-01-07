#!/usr/bin/env python3

"""
Calculate sum of first 10 primes.

https://oeis.org/A007504

0, 2, 5, 10, 17, 28, 41, 58, 77, 100, 129, 160, 197, 238
                                      ^^^
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
def sum_10_primes() -> int:
    primes: auto = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    p: auto = 2
    for i in range(10):
        while True:
            if is_prime(p):
                break
            p += 1
        primes[i] = p
        p += 1
    print(primes)
    return sum(primes)

if __name__ == '__main__':
    import sys
    try:
        ...
    except IndexError:
        print(f'usage: {sys.argv[0]} N')
        sys.exit(1)

    print(f'{sum_10_primes() = }')

