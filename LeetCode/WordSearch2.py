class Solution(object):
    def navigate(self, board, words, prefix, row, col, curr_str, num_rows, num_cols, start_cache, idx_cache):
        out = set()

        q = str(row) + "__" + str(col)

        if row < 0 or row >= num_rows or col < 0 or col >= num_cols or q in idx_cache:
            return out

        idx_cache = set(list(idx_cache) + [q])

        if curr_str in words:
            out.add(curr_str)

        if col < num_cols - 1:
            if curr_str + board[row][col + 1] in prefix:
                out.update(self.navigate(board, words, prefix, row, col + 1, curr_str + board[row][col + 1], num_rows, num_cols,
                                         start_cache, idx_cache))

            if str(row) + "__" + str(col + 1) not in start_cache:
                start_cache.add(str(row) + "__" + str(col + 1))
                out.update(self.navigate(board, words, prefix, row, col + 1, board[row][col + 1], num_rows, num_cols,
                                         start_cache, set()))


        if row < num_rows - 1:
            if curr_str + board[row + 1][col] in prefix:
                out.update(self.navigate(board, words, prefix, row + 1, col, curr_str + board[row + 1][col], num_rows, num_cols,
                                         start_cache, idx_cache))

            if str(row + 1) + "__" + str(col) not in start_cache:
                start_cache.add(str(row + 1) + "__" + str(col))
                out.update(self.navigate(board, words, prefix, row + 1, col, board[row + 1][col], num_rows, num_cols,
                                         start_cache, set()))


        if col > 0:
            if curr_str + board[row][col - 1] in prefix:
                out.update(self.navigate(board, words, prefix, row, col - 1, curr_str + board[row][col - 1], num_rows, num_cols,
                                         start_cache, idx_cache))

            if str(row) + "__" + str(col - 1) not in start_cache:
                start_cache.add(str(row) + "__" + str(col - 1))
                out.update(self.navigate(board, words, prefix, row, col - 1, board[row][col - 1], num_rows, num_cols,
                                         start_cache, set()))


        if row > 0:
            if curr_str + board[row - 1][col] in prefix:
                out.update(self.navigate(board, words, prefix, row - 1, col, curr_str + board[row - 1][col], num_rows, num_cols,
                                         start_cache, idx_cache))

            if str(row - 1) + "__" + str(col) not in start_cache:
                start_cache.add(str(row - 1) + "__" + str(col))
                out.update(self.navigate(board, words, prefix, row - 1, col, board[row - 1][col], num_rows, num_cols,
                                         start_cache, set()))


        return out

    def findWords(self, board, words):
        if len(board) == 0:
            return []

        prefix = dict()

        for word in words:
            for pos in range(len(word)):
                if word[:pos + 1] not in prefix:
                    prefix[word[:pos + 1]] = []
                prefix[word[:pos + 1]].append(word)

        x, y = len(board), len(board[0])

        start_cache, idx_cache = set(["0__0"]), set()

        return list(self.navigate(board, set(words), prefix, 0, 0, board[0][0], x, y, start_cache, idx_cache))


sol = Solution()

board = [["b"],
         ["a"],
         ["b"],
         ["b"],
         ["a"]]

words = ["baa","abba","baab","aba"]

print(sol.findWords(board, words))