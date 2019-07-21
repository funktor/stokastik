class Solution(object):
    def minSwap(self, A, B):
        swaps = [(0, 0)]*len(A)
        
        for i in range(len(A)):
            if i == 0:
                swaps[0] = (0, 1) if A[i] != B[i] else (0, 0)
            else:
                if A[i] > A[i-1] and B[i] > B[i-1] and A[i] > B[i-1] and B[i] > A[i-1]:
                    a = min(swaps[i-1][0], swaps[i-1][1])
                elif A[i] > A[i-1] and B[i] > B[i-1]:
                    a = swaps[i-1][0]
                else:
                    a = swaps[i-1][1]
                    
                if B[i] > A[i-1] and A[i] > B[i-1] and B[i] > B[i-1] and A[i] > A[i-1]:
                    b = min(swaps[i-1][0], swaps[i-1][1])
                elif B[i] > A[i-1] and A[i] > B[i-1]:
                    b = swaps[i-1][0]
                else:
                    b = swaps[i-1][1]
                
                swaps[i] = (a, b+1)
                
        return min(swaps[-1][0], swaps[-1][1])
