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

## Adding `cboost`

```python
import cboost               # <-- import

@cboost.make_cpp            # <-- decorator
def fibo(n: int) -> int:
    if n in (0, 1):
        return n
    return fibo(n-2) + fibo(n-1)
```

## Generated C/C++ code:
```cpp
// Module translated by cboost
extern "C" {
long fibo(long n);
}
long fibo(long n)
{
    if ((n == 0) || (n == 1)){
        return n;
    }
    return fibo(n - 2) + fibo(n - 1);
}

// End of module
```

## Example benchmarks:

Without `cboost`:
```
$ time CBOOST_DISABLE=1 ./example_primes.py 1000000
Warning: cboost disabled by CBOOST_DISABLE
how_many_primes(n) = 78498

real	0m8.506s
user	0m8.494s
sys	0m0.009s
```

With `cboost`:
```
$ time ./example_primes.py 1000000
how_many_primes(n) = 78498

real	0m0.667s
user	0m0.647s
sys	0m0.020s
```

Testing environment:
- CPU: **Intel(R) Core(TM) i5 CPU M 580  @ 2.67GHz**
- Python: **3.10.1**
- OS: **Ubuntu 20.04.3 LTS**
- Kernel: **Linux 5.4.0-91-generic**
- gcc: **g++ (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0**

