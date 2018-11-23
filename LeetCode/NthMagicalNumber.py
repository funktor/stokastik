class Solution(object):
    def gcd(self, a, b):
        if a == 0:
            return b
        else:
            return self.gcd(b % a, a)
        
    def nthMagicalNumber(self, N, A, B):
        gcd = self.gcd(A, B)
        lcm, lower, higher = (A*B)/gcd, min(A, B), max(A, B)
        
        mod = 10**9 + 7
        left, right = 1, N
        
        while left <= right:
            mid = (left + right)/2
            min_val = lower*mid
            alt_val = int(min_val/higher)*higher
            
            x1, y1, z1 = int(min_val/A), int(min_val/B), int(min_val/lcm)
            x2, y2, z2 = int(alt_val/A), int(alt_val/B), int(alt_val/lcm)
            
            a, b = x1 + y1 - z1, x2 + y2 - z2
            
            if a == N or b == N:
                return min_val%mod if a == N else alt_val%mod
            elif a < N:
                left = mid + 1
            else:
                right = mid - 1
            
        min_val = lower*left
        alt_val = int(min_val/higher)*higher
        
        return min(min_val, alt_val)%mod