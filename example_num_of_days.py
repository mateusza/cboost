#!/usr/bin/env python3

"""
Show number of days of given month of the year.

https://en.wikipedia.org/wiki/Leap_year#Gregorian_calendar
"""

import cboost

@cboost.make_cpp
def is_leap(year: int) -> bool:
    return (year % 400 == 0) or (year % 100 != 0 and year % 4 == 0)

@cboost.make_cpp
def num_of_days(year: int, month: int) -> int:
    days: 'auto' = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    return days[month] + (1 if month == 2 and is_leap(year) else 0)

if __name__ == '__main__':
    import sys
    try:
        y, m = int(sys.argv[1]), int(sys.argv[2])
    except IndexError:
        print(f'usage: {sys.argv[0]} YEAR MONTH')
        sys.exit(1)

    print(f'{num_of_days(y, m) = }')

