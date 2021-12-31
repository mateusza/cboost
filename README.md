# `cboost`

Boost your **python** code by automatically converting it to **C/C++** on the fly.

Everything happens automagically just by adding single `import` and a decorator.

Read more about [current status, features, and bugs](TODO.md)...

## Example python code to recursively calculate *n-th* term of Fibonacci sequence.

```python
def fibo(n: int) -> int:
    if n in (0, 1):
        return n
    return fibo(n-2) + fibo(n-1)
```

## With `cboost`

```python
import cboost

@cboost.make_c
def fibo(n: int) -> int:
    if n in (0, 1):
        return n
    return fibo(n-2) + fibo(n-1)
```

## Generated C/C++ code:
```cpp
extern "C" {
long fibo (long n);
}
long fibo (long n){
    if((n == 0) || (n == 1)){
        return n;
    }
return (fibo((n - 2)) + fibo((n - 1)));
}
```


