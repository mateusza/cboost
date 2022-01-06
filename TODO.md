# Status and TODO

## Bugs

Everything is buggy. You are lucky if it worked with your code.

- cboost handles lists in a fancy way, and sometimes does the opposite of what python does:
    1. `a = [1, 2, 3, 4, 5]; b = a;` this will make another **copy** of the list (vector)

## Currently supported:
- `if` condition
- `while` loop
- `for` loop with either `range()` or a list
- `int` (`long`) as return and argument type only
- types:
    - `str` (`std::string`)
    - `list` (`std::vector`)
- functions:
    - `len()`
    - `any()`, `all()`
    - `sum()`
    - `abs()`
- methods:
    - `.join()`
    - `.split()`
- some arithmetic and boolean operators
- `list` and `str` multiplication
- only variables with explicitly given types (`a: int = 7`) or `auto`

## Todo:
- argument type mapping and support
- better `array`/`list` support
- pythonic loop control `else:`
- missing arithmetic, bitwise, logical operators
- python functions like:
 - `min()`
 - `max()`
- python list expressions `[x for x in ...]` etc
- classes and methods
- ability to select compiler, flags etc

## Things that accidentially work better

**cboost** does some things that python doesn't. It's not yet decided whether we will break these in future or leave as a feature.

- `", ".join([1, 3, 4])` automatically converts list values to strings, while python raises `TypeError: sequence item 0: expected str instance, int found`
- `sum(["aaa", "bbb"])` concatenates strings, while python raises `TypeError: unsupported operand type(s) for +: 'int' and 'str'`
