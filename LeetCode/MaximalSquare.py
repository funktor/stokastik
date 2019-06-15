class Solution(object):
    def max_continuous(self, arr):
        max_c, curr_c = 0, 0
        for i in range(len(arr)):
            if arr[i] == "1":
                curr_c += 1
            else:
                max_c = max(max_c, curr_c)
                curr_c = 0
        max_c = max(max_c, curr_c)
        return max_c
    
    def maximalSquare(self, matrix):
        if len(matrix) == 0:
            return 0
        
        n = min(len(matrix), len(matrix[0]))+1
        
        max_n1 = -float("Inf")
        for i in range(len(matrix)):
            max_n1 = max(max_n1, self.max_continuous(matrix[i]))
            
        matrix = map(list, zip(*matrix))
        
        max_n2 = -float("Inf")
        for i in range(len(matrix)):
            max_n2 = max(max_n2, self.max_continuous(matrix[i]))
            
        matrix = map(list, zip(*matrix))
            
        cached = [[False]*len(matrix[0]) for i in range(len(matrix))]
        
        max_length = 0
        for length in range(1, min(max_n1, max_n2)+1):
            for i in range(len(matrix)-length+1):
                for j in range(len(matrix[0])-length+1):
                    if length == 1:
                        cached[i][j] = True if matrix[i][j] == "1" else False
                    else:
                        a = cached[i][j]
                        b = i+1 < len(matrix) and cached[i+1][j]
                        c = j+1 < len(matrix[0]) and cached[i][j+1]
                        d = i+1 < len(matrix) and j+1 < len(matrix[0]) and cached[i+1][j+1]

                        cached[i][j] = a and b and c and d
                        
                    if cached[i][j]:
                        max_length = max(max_length, length)
        
        return max_length**2
