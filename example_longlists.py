#!/usr/bin/env python3

"""
List operations
"""

import cboost

@cboost.make_cpp
def test_lists():
    dlugosc: int = 1024*1024*1
    lista1: 'auto' = [0]*dlugosc
    lista2: 'auto' = [0]*dlugosc
    lista3: 'auto' = [0.1]

    for i in range(dlugosc):
        lista1[i] = i
        lista2[i] = i % 8

    for i in range(dlugosc):
        lista3.append(1111.1 / (1+lista1[i]) + lista2[i])

    print(sum(lista3))
    print(len(lista3))

if __name__ == '__main__':
    import sys
    try:
        ...
    except IndexError:
        print(f'usage: {sys.argv[0]}')
        sys.exit(1)

    test_lists()
