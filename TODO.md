# Status and TODO

## Bugs

Everything is buggy. You are lucky if it worked with your code.

- 

## Currently supported:
- `if` condition
- `while` loop
- `for` loop only when iterating over `range()` in incrementing order
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
- only variables with explicitly given types (`a: int = 7`)

## Todo:
- argument type mapping and support
- better `array`/`list` support
- pythonic loop control `else:`
- missing arithmetic, bitwise, logical operators
- python functions like:
 - `min()`
 - `max()`
- python list expressions `[x for x in ...]`
- classes and methods
- ability to select compiler, flags etc

