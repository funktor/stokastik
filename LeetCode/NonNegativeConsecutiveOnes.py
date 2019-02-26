class Solution(object):
    def findIntegers(self, num):
        bin_str = str(bin(num))[2:]
        n = len(bin_str)
        
        full_ints = [0]*(n+1)
        full_ints[0], full_ints[1] = 1, 2
        
        for i in range(2, n+1):
            full_ints[i] = full_ints[i-1] + full_ints[i-2]
            
        cnt = 0
        for i in reversed(range(n)):
            if i == n-1:
                cnt = 2 if bin_str[i] == '1' else 1
            else:
                if bin_str[i] == '1':
                    a = full_ints[n-i-2] if bin_str[i+1] == '1' else cnt
                    cnt = full_ints[n-i-1] + a
        
        return cnt
