#!/usr/bin/env python3.10

import ast
import cboost

m1 = ast.parse("a = 123456")
m2 = cboost.convert(m1)

print(m2)
