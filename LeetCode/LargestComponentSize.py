import numpy as np, collections, time
        
class Solution(object):
    def sieve(self, n):
        arr = [True]*(n+1)
        arr[0], arr[1] = False, False
        
        sqrt_n = int(np.sqrt(n))
        for i in range(2, sqrt_n+1):
            if arr[i]:
                j = i**2
                while j <= n:
                    arr[j] = False
                    j += i
                    
        primes = []
        for i in range(2, n+1): 
            if arr[i]:
                primes.append(i)
        
        return arr, primes
    
    def get_prime_factors(self, n, primes, prime_arr):
        factors = set()
        
        if prime_arr[n] is False:
            i = 0
            while primes[i] <= np.sqrt(n):
                if n % primes[i] == 0:
                    factors.add(primes[i])
                    n /= primes[i]
                else:
                    i += 1
                
        factors.add(n)

        return factors
    
    def bfs_search(self, elem_to_factors, factor_to_elem, start, visited):
        queue, num_connected = collections.deque([start]), 0
        factor_added = set()
        
        visited.add(start)
        
        while len(queue) > 0:
            pt = queue.popleft()
            num_connected += 1
            
            for factor in elem_to_factors[pt]:
                if factor not in factor_added:
                    factor_added.add(factor)
                    elems = factor_to_elem[factor]
                    for x in elems:
                        if x not in visited:
                            queue.append(x)
                            visited.add(x)
                        
        return num_connected
        
    def largestComponentSize(self, A):
        n = len(A)
        prime_arr, primes = self.sieve(np.max(A))
        factor_to_elem, elem_to_factors = collections.defaultdict(list), []
        
        for i in range(n):
            prime_factors = self.get_prime_factors(A[i], primes, prime_arr)
            elem_to_factors.append(prime_factors)
            
            for y in prime_factors:
                factor_to_elem[y].append(i)
        
        max_length, visited = 0, set()

        for i in range(n):
            if i not in visited:
                path_len = self.bfs_search(elem_to_factors, factor_to_elem, i, visited)
                max_length = max(max_length, path_len)

                m = n-len(visited)

                if max_length > m:
                    break
        
        return max_length
