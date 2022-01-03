#!/usr/bin/env python3.10

"""
Create string by repeating literan n times and return number of chars.
"""

import cboost

@cboost.make_cpp
def echo(n: int) -> int:
    napis: str = ""
    for i in range(n):
        napis += "echo "
    print("Hello: " + napis)
    print("10 * \"kot\"")
    print(10 * "kot")
    print('"pies" * 10')
    print("pies" * 10)
    print(f'liczba: {123} napis: {napis}')
    return len(napis)

if __name__ == '__main__':
    import sys
    try:
        n = int(sys.argv[1])
    except IndexError:
        print(f'usage: {sys.argv[0]} N')
        sys.exit(1)

    print(f'{echo(n) = }')

