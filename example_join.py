#!/usr/bin/env python3.10

"""
Create string by repeating literan n times and return number of chars.
"""

import cboost

@cboost.make_cpp
def echo():
    litery: str = "abcdefghijklmno"
    napis2: str = ";".join(litery)

    miesiace: list = ["styczen", "luty", "marzec"]
    print("litery: " + napis2)
    print("miesiace: " + ", ".join(miesiace))

    for imie in "Ala Basia Czesia Daria Eryka".split(" "):
        print(f"{imie = }")

if __name__ == '__main__':
    import sys
    try:
        ...
    except IndexError:
        print(f'usage: {sys.argv[0]}')
        sys.exit(1)

    echo()
