from libc.stdlib cimport malloc, free
from libc.math cimport sqrt

def sieve(int n):
    cdef int *arr = <int *> malloc((n+1) * sizeof(int))
    cdef int sqrt_n, i, j
    
    for i in range(2, n+1):
        arr[i] = 1

    sqrt_n = int(sqrt(n))
    for i in range(2, sqrt_n+1):
        if arr[i] == 1:
            j = i**2
            while j <= n:
                arr[j] = 0
                j += i

    cdef list out = [i for i in range(2, n+1) if arr[i] == 1]
    free(arr)
    return out