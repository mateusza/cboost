_If it looks stupid but works, it isn't stupid_

# cboost

Boost your **python** code by automatically converting it to **C/C++** on the fly.

Everything happens automagically just by adding single `import` and a decorator.

Read more about [current status, features, and bugs](TODO.md)...

## Example python code to [count number of primes](https://en.wikipedia.org/wiki/Prime-counting_function) less than *n*:

```python
def is_prime(n: int) -> bool:
    if n in (0, 1):
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    for d in range(3, n, 2):
        if n % d == 0:
            return False
        if d * d > n:
            break
    return True

def how_many_primes(limit: int) -> int:
    how_many: int = 0
    for i in range(2, limit):
        if is_prime(i):
            how_many += 1
    return how_many

if __name__ == '__main__':
    import sys
    try:
        n = int(sys.argv[1])
    except IndexError:
        print(f'usage: {sys.argv[0]} N')
        sys.exit(1)

    print(f'{how_many_primes(n) = }')
```

## Using cboost

Just `import cboost` and decorate each function with `cboost.make_cpp`.

```python
import cboost                           # <-- import

@cboost.make_cpp                        # <-- decorator
def is_prime(n: int) -> bool:
    ...

@cboost.make_cpp                        # <-- decorator
def how_many_primes(limit: int) -> int:
    ...

...
```

### Not using cboost

You can temporarily disable `cboost` by calling `cboost.disable()` before calling decorated function.

You can also `export CBOOST_DISABLE=1` before running the script.

## Generated C/C++ code:
```cpp
// Module translated by cboost
extern "C" {
int is_prime(long n);
long how_many_primes(long limit);
}
int is_prime(long n)
{
    if ((n == 0) || (n == 1)){
        return 0;
    }
    if ((n == 2) || (n == 3)){
        return 1;
    }
    if ((n % 2) == 0){
        return 0;
    }
    /* This is translated from something else (eg. range()): */
    for (auto d = 3; d < n; d += 2){
        if ((n % d) == 0){
            return 0;
        }
        if ((d * d) > n){
            break;
        }
    }
    /* Hope it works */
    return 1;
}

long how_many_primes(long limit)
{
    int how_many = 0;
    /* This is translated from something else (eg. range()): */
    for (auto i = 2; i < limit; ++i){
        if (is_prime(i)){
            how_many += 1;
        }
    }
    /* Hope it works */
    return how_many;
}

// End of module
```

## Example benchmarks:

Without `cboost`:
```
$ time CBOOST_DISABLE=1 ./example_primes.py 10000000
Warning: cboost disabled by CBOOST_DISABLE
how_many_primes(n) = 664579

real	1m59.170s
user	1m58.989s
sys     0m0.112s
```

With `cboost`:
```
$ time ./example_primes.py 10000000
how_many_primes(n) = 664579

real	0m7.972s
user	0m7.957s
sys     0m0.012s
```

This simple example gives almost **15 times** performance boost.

Testing environment:
- CPU: **Intel(R) Core(TM) i5 CPU M 580  @ 2.67GHz**
- Python: **3.10.1**
- OS: **Ubuntu 20.04.3 LTS**
- Kernel: **Linux 5.4.0-91-generic**
- gcc: **g++ (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0**

