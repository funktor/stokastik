class Solution(object):
    def turbulent(self, A, i):
        if i == len(A)-1:
            return 1, 1
        
        t1, t2 = self.turbulent(A, i+1)
        
        a = i+2 < len(A) and (A[i] < A[i+1] > A[i+2] or A[i] > A[i+1] < A[i+2])
        b = i+2 >= len(A) and (A[i] < A[i+1] or A[i] > A[i+1])
        
        if a or b:
            t1 += 1
        else:
            t1 = 2 if (A[i] < A[i+1] or A[i] > A[i+1]) else 1
        
        if t1 > t2:
            return t1, t1
        
        return t1, t2
            
    def maxTurbulenceSize(self, A):
        out = self.turbulent(A, 0)
        return out[1]
