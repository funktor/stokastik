class NumMatrix(object):

    def __init__(self, matrix):
        self.mat_sum = []
        for i in range(len(matrix)):
            self.mat_sum.append([0]*len(matrix[0]))
        
        for i in range(len(matrix)):
            curr_row_sum = 0
            for j in range(len(matrix[0])):
                curr_row_sum += matrix[i][j]
                if i == 0:
                    self.mat_sum[i][j] = curr_row_sum
                else:
                    self.mat_sum[i][j] = self.mat_sum[i-1][j] + curr_row_sum

    def sumRegion(self, row1, col1, row2, col2):
        a = self.mat_sum[row2][col2]
        b = self.mat_sum[row1-1][col2] if row1 > 0 else 0
        c = self.mat_sum[row2][col1-1] if col1 > 0 else 0
        d = self.mat_sum[row1-1][col1-1] if row1 > 0 and col1 > 0 else 0
        return a - b - c + d

# Your NumMatrix object will be instantiated and called as such:
# obj = NumMatrix(matrix)
# param_1 = obj.sumRegion(row1,col1,row2,col2)
