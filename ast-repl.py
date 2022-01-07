#!/usr/bin/env python3

import sys
import ast

if sys.version_info > (3, 9):
    opts = {'indent': 4}
else:
    opts = {}

while True:
    code = input(">>> ")
    d = ast.dump(ast.parse(code), **opts)
    print(f'{d}\n\n')

