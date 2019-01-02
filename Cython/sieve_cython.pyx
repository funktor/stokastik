#cython: boundscheck=False, wraparound=False, nonecheck=False

from libc.math cimport sqrt
import numpy as np

def sieve(int n):
    np_arr = np.empty(n+1)
    np_arr.fill(1)
    
    cdef int[:] arr = np_arr
    cdef int sqrt_n, i, j
    cdef list primes
    
    arr[0], arr[1] = 0, 0

    sqrt_n = int(sqrt(n))
    for i in range(2, sqrt_n+1):
        if arr[i] == 1:
            j = i**2
            while j <= n:
                arr[j] = 0
                j += i

    primes = []
    for i in range(2, n+1): 
        if arr[i] == 1:
            primes.append(i)

    return primes