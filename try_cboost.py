#!/usr/bin/env python3.10

import ast
import cboost

code = """
a: int = 123
b = 546
"""

m1 = ast.parse(code)
m2 = cboost.convert(m1)

print(m2)
print(ast.dump(m1, indent=4))
print(cboost.dump(m2, indent=4))
