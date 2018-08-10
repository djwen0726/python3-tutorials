import math


def func_get_prime(n):
    return filter(lambda x: not [x % i for i in range(2, int(math.sqrt(x)) + 1) if x % i == 0], range(2, n + 1))


print(list(func_get_prime(1000000)))
