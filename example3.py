#!/usr/bin/env python3.10

import sys

def prod_of_range(a: int, b: int, m: int) -> int:
    result: int = 1
    for i in range(a, b+1):
        result *= i
        result %= m
    return result

def max_of_ranges(size: int) -> int:
    modulo: int = 777777
    the_max: int = 0
    for i in range(1, size):
        for j in range(1, size):
            p: int = prod_of_range(i, j, modulo)
            if p > the_max:
                the_max = p
    return the_max

try:
    s = int(sys.argv[1])
except IndexError:
    print(f'usage: {sys.argv[0]} NUM')
    sys.exit(1)

print(max_of_ranges(s))
