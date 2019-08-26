class Solution(object):
    def largest1BorderedSquare(self, grid):
        max_con_ones_row = -float("Inf")
        for i in range(len(grid)):
            curr_ones = 0
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    curr_ones += 1
                else:
                    max_con_ones_row = max(max_con_ones_row, curr_ones)
                    curr_ones = 0
        
            max_con_ones_row = max(max_con_ones_row, curr_ones)
        
        max_con_ones_col = -float("Inf")
        for j in range(len(grid[0])):
            curr_ones = 0
            for i in range(len(grid)):
                if grid[i][j] == 1:
                    curr_ones += 1
                else:
                    max_con_ones_col = max(max_con_ones_col, curr_ones)
                    curr_ones = 0
        
            max_con_ones_col = max(max_con_ones_col, curr_ones)
        
        K = min(max_con_ones_row, max_con_ones_col)
        
        row_rle = [[[False]*(K+1) for j in range(len(grid[0]))] for i in range(len(grid))]
        col_rle = [[[False]*(K+1) for j in range(len(grid[0]))] for i in range(len(grid))]
        
        output = 0
        
        for k in range(1, K+1):
            for i in range(len(grid)):
                for j in range(len(grid[0])):
                    if k == 1:
                        row_rle[i][j][k] = grid[i][j] == 1
                        col_rle[i][j][k] = grid[i][j] == 1
                    else:
                        row_rle[i][j][k] = j+1 < len(grid[0]) and row_rle[i][j+1][k-1] and grid[i][j] == 1
                        col_rle[i][j][k] = i+1 < len(grid) and col_rle[i+1][j][k-1] and grid[i][j] == 1
                        
                    out = (k == 1 and grid[i][j] == 1) or (i+1 < len(grid) and i+k-1 < len(grid) and j+1 < len(grid[0]) and j+k-1 < len(grid[0]) and grid[i][j] == 1 and row_rle[i][j+1][k-1] and row_rle[i+k-1][j+1][k-1] and col_rle[i+1][j][k-1] and col_rle[i+1][j+k-1][k-1])
                    
                    if out:
                        output = max(output, k*k)
        
        return output
