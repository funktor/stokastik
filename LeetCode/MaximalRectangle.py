class Solution(object):
    def maximalRectangle(self, matrix):
        if len(matrix) == 0:
            return 0

        max_len_rt = [[0] * (len(matrix[0]) + 1) for _ in range(len(matrix) + 1)]

        max_area = 0

        for row in reversed(range(len(matrix))):
            for col in reversed(range(len(matrix[0]))):

                if matrix[row][col] == "1":
                    x = max_len_rt[row][col + 1] + 1
                    max_len_rt[row][col] = x

                    max_area = max(max_area, x)

                    for row2 in range(row + 1, len(matrix)):
                        y = max_len_rt[row2][col]

                        if y <= x:
                            area = (row2 - row + 1) * y
                            x = y
                        else:
                            area = (row2 - row + 1) * x

                        max_area = max(max_area, area)

        return max_area