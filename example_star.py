#!/usr/bin/env python3.10

"""
List operations
"""

import cboost

@cboost.make_cpp
def test_lists():
    lista1: 'auto' = [5,6,7,8,9]
    lista2: 'auto' = [10,11,12]
    lol: 'auto' = 123
    lista3: 'auto' = [0,1,2,*lista1,lol,99,*lista2,*[99999,1000000]]
    lista3[0] = 1111

    print('-'.join(lista3))

if __name__ == '__main__':
    import sys
    try:
        ...
    except IndexError:
        print(f'usage: {sys.argv[0]}')
        sys.exit(1)

    test_lists()
