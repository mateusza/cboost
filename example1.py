#!/usr/bin/env python3.10

import cboost
import timeit
import sys

def fibo_py(n: int) -> int:
    if n in (0, 1):
        return n
    return fibo_py(n-2) + fibo_py(n-1)

@cboost.make_c
def fibo(n: int) -> int:
    if n in (0, 1):
        return n
    return fibo(n-2) + fibo(n-1)

# optional, if not used, it will compile on first use
cboost.compile_c()

print()
for i in range(10):
    print(f"{i = }")
    funcs = [fibo_py, fibo]
    for f in funcs:
        v, t = f(i), timeit.timeit(lambda: f(i))
        print(f"fibo({i}) = {v}   [in {t:.2f}s]  using {f}")
