#!/usr/bin/env python3.10

import cboost

@cboost.make_c
def prod_of_range(a: int, b: int, m: int) -> int:
    result: int = 1
    for i in range(a, b+1):
        result *= i
        result %= m
    return result

@cboost.make_c
def max_of_ranges() -> int:
    modulo: int = 777777
    the_max: int = 0
    for i in range(1, 600):
        for j in range(1, 600):
            p: int = prod_of_range(i, j, modulo)
            if p > the_max:
                the_max = p
    return the_max

print(max_of_ranges())

