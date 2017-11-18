class Solution(object):
    def capture(self, board, num_rows, num_cols, components):

        for component in components:
            flag = True

            for point in component:
                if point[0] == 0 or point[0] == num_rows-1 or point[1] == 0 or point[1] == num_cols-1:
                    flag = False
                    break

            if flag is True:
                for point in component:
                    board[point[0]][point[1]] = 'X'


    def connected_components(self, board, num_rows, num_cols):
        components = []
        visited = set()

        row = 0
        while row < len(board):
            col = 0
            while col < len(board[0]):
                if str(row) + "__" + str(col) not in visited:
                    if board[row][col] == 'O':
                        component = []
                        stack = [(row,col)]
                        visited.add(str(row) + "__" + str(col))

                        while len(stack) > 0:
                            a = stack.pop()
                            component.append(a)

                            i, j = a[0], a[1]

                            if j + 1 < num_cols and board[i][j + 1] == 'O' and str(i) + "__" + str(j + 1) not in visited:
                                visited.add(str(i) + "__" + str(j + 1))
                                stack.append((i, j + 1))
                            if i + 1 < num_rows and board[i + 1][j] == 'O' and str(i + 1) + "__" + str(j) not in visited:
                                visited.add(str(i + 1) + "__" + str(j))
                                stack.append((i + 1, j))
                            if j - 1 >= 0 and board[i][j - 1] == 'O' and str(i) + "__" + str(j - 1) not in visited:
                                visited.add(str(i) + "__" + str(j - 1))
                                stack.append((i, j - 1))
                            if i - 1 >= 0 and board[i - 1][j] == 'O' and str(i - 1) + "__" + str(j) not in visited:
                                visited.add(str(i - 1) + "__" + str(j))
                                stack.append((i - 1, j))

                        components.append(component)
                col += 1
            row += 1

        return components


    def solve(self, board):
        if len(board) == 0:
            board = []
        else:
            num_rows, num_cols = len(board), len(board[0])
            components = self.connected_components(board, num_rows, num_cols)
            self.capture(board, num_rows, num_cols, components)


sol = Solution()
board = [["X","X","X","X"],
         ["X","O","O","X"],
         ["X","X","O","X"],
         ["X","O","X","X"]]

sol.solve(board)
print board