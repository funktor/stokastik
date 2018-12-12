class Solution(object):
    def get_smaller(self, A):
        smaller = range(len(A))
        
        for i in reversed(range(len(A))):
            if i == len(A)-1:
                smaller[i] = i+1
            else:
                y = i+1
                while A[y] >= A[i]:
                    y = smaller[y]
                    if y == len(A):
                        break
                smaller[i] = y
                
        return smaller
    
    def sumSubarrayMins(self, A):
        smaller = self.get_smaller(A)
        sums, m = [0]*(len(A)+1), 10**9+7
        
        for i in reversed(range(len(A))):
            x = smaller[i]
            sums[i] = (A[i]*(x-i) + sums[x])%m
        
        out = 0
        for i in range(len(A)):
            out += sums[i]%m
        
        return out%m
