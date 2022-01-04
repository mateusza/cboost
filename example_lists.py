#!/usr/bin/env python3.10

"""
List operations
"""

import cboost

@cboost.make_cpp
def test_lists():
    lista1: 'auto' = [5,6,7,8,9]
    lista2: 'auto' = [10,11,12]

    lista1.append(lista2[1])
    lista1.extend([5,6,7])

    lista1.extend([0,1]*7)
    lista1.extend(3*[3,3,3])

    for e in lista1 + lista2:
        print(e)
    print(lista1 + lista2)

if __name__ == '__main__':
    import sys
    try:
        ...
    except IndexError:
        print(f'usage: {sys.argv[0]}')
        sys.exit(1)

    test_lists()
