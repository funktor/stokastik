class Solution(object):
    def get_max_flow(self, arr):
        flows = [-1] * len(arr)

        for idx in reversed(range(len(arr))):
            if idx == len(arr) - 1 or arr[idx] > arr[idx + 1]:
                flows[idx] = idx + 1
            else:
                y = idx + 1

                while True:
                    x = flows[y]

                    if x == len(arr) or arr[idx] > arr[x]:
                        flows[idx] = x
                        break

                    else:
                        y = x

        return flows

    def get_column_flow(self, matrix, max_len_rt, direction=1):
        flow = [[0] * (len(matrix[0])) for _ in range(len(matrix))]

        n, m = range(len(matrix)), range(len(matrix[0]))

        r_iter = n[::-1] if direction == 1 else n
        c_iter = m[::-1] if direction == 1 else m

        edge_row = len(matrix) - 1 if direction == 1 else 0

        for row in r_iter:
            for col in c_iter:
                if matrix[row][col] == "1":
                    if row == edge_row or max_len_rt[row][col] > max_len_rt[row + direction][col]:
                        flow[row][col] = row + direction

                    else:
                        y = row + direction

                        while True:
                            x = flow[y][col]

                            if x == edge_row + direction or max_len_rt[row][col] > max_len_rt[x][col]:
                                flow[row][col] = x
                                break

                            else:
                                y = x

        return flow

    def maximalRectangle(self, matrix):
        if len(matrix) == 0:
            return 0

        max_len_rt = [[0] * (len(matrix[0]) + 1) for _ in range(len(matrix) + 1)]

        for row in reversed(range(len(matrix))):
            for col in reversed(range(len(matrix[0]))):
                if matrix[row][col] == "1":
                    max_len_rt[row][col] = max_len_rt[row][col + 1] + 1

        fwd_flow = self.get_column_flow(matrix, max_len_rt, 1)
        bwd_flow = self.get_column_flow(matrix, max_len_rt, -1)

        max_area = 0

        for row in range(len(matrix)):
            for col in range(len(matrix[0])):
                if matrix[row][col] == "1":
                    width = max_len_rt[row][col]
                    a = fwd_flow[row][col] - row
                    b = row - bwd_flow[row][col]

                    height = a + b - 1

                    max_area = max(max_area, width * height)

        return max_area