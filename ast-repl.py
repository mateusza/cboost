#!/usr/bin/env python3.10

import sys
import ast

while True:
    code = input(">>> ")
    d = ast.dump(ast.parse(code), indent=4)
    print(f'{d}\n\n')

