class Solution(object):
    def mctFromLeafValues(self, arr):
        cache = [[0]*len(arr) for i in range(len(arr))]
        
        for length in range(1, len(arr)+1):
            for i in range(len(arr)-length+1):
                j = i + length - 1
                if length == 1:
                    cache[i][j] = (0, arr[i])
                
                elif length == 2:
                    cache[i][j] = (arr[i]*arr[j], max(arr[i], arr[j]))
                
                else:
                    min_sum = float("Inf")
                    
                    for k in range(i, j):
                        a1, b1 = cache[i][k]
                        a2, b2 = cache[k+1][j]
                        
                        new_sum = a1 + a2 + b1*b2
                        
                        if new_sum < min_sum:
                            min_sum = new_sum
                            cache[i][j] = (new_sum, max(b1, b2))
        
        return cache[0][len(arr)-1][0]
    
    
