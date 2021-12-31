#!/usr/bin/env python3.10

import cboost
import timeit
import sys

def tribo_py(n: int) -> int:
    if n in (0, ):
        return 0
    if n in (1, 2):
        return 1
    return tribo_py(n-3) + tribo_py(n-2) + tribo_py(n-1)

@cboost.make_c
def tribo(n: int) -> int:
    if n in (0, ):
        return 0
    if n in (1, 2):
        return 1
    return tribo(n-3) + tribo(n-2) + tribo(n-1)

tribo(0)

print()
for i in range(10):
    print(f"{i = }")
    funcs = [tribo_py, tribo]
    for f in funcs:
        v, t = f(i), timeit.timeit(lambda: f(i))
        print(f"tribo({i}) = {v}   [in {t:.2f}s]  using {f}")
