#!/usr/bin/env python3.10

import ast
import cboost
import inspect

def is_prime(n: int) -> bool:
    if n == 0 or n == 1:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    for d in range(3, n):
        if n % d == 0:
            return False
        if d * d > n:
            break
    return True

code = inspect.getsource(is_prime)

m1 = ast.parse(code)
print(ast.dump(m1, indent=4))

m2 = cboost.convert(m1)
print(m2)
print(cboost.dump(m2, indent=4))

print(cboost.render(m2, indent=4))
