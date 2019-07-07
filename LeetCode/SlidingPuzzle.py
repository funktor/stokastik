class Solution(object):
    def swap(self, board, x1, y1, x2, y2, heap, cache, steps, cost):
        if 0 <= x2 < len(board) and 0 <= y2 < len(board[0]):
            a1 = 1 if board[x1][y1] != (3*x1+y1+1) % 6 else 0
            b1 = 1 if board[x2][y2] != (3*x2+y2+1) % 6 else 0

            temp = board[x1][y1]
            board[x1][y1] = board[x2][y2]
            board[x2][y2] = temp

            a2 = 1 if board[x1][y1] != (3*x1+y1+1) % 6 else 0
            b2 = 1 if board[x2][y2] != (3*x2+y2+1) % 6 else 0

            new_cost = cost - (a1 + b1) + (a2 + b2)
            q = tuple([tuple(x) for x in board])

            if q not in cache or steps+1 < cache[q]:
                heapq.heappush(heap, (new_cost, x2, y2, steps+1, board))
                cache[q] = steps+1
                
                    
    def slidingPuzzle(self, board):
        cost = 0
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 0:
                    start = (i, j)
                cost += 1 if board[i][j] != (3*i+j+1) % 6 else 0
        
        heap = [(cost, start[0], start[1], 0, board)]
        cache = {tuple([tuple(x) for x in board]):cost}
        
        min_steps = float("Inf")
        
        while len(heap) > 0:
            cost, x, y, steps, board = heapq.heappop(heap)
            
            if cost == 0:
                min_steps = min(min_steps, steps)
            
            self.swap([z[:] for z in board], x, y, x+1, y, heap, cache, steps, cost)
            self.swap([z[:] for z in board], x, y, x-1, y, heap, cache, steps, cost)
            self.swap([z[:] for z in board], x, y, x, y+1, heap, cache, steps, cost)
            self.swap([z[:] for z in board], x, y, x, y-1, heap, cache, steps, cost)
        
        return min_steps if min_steps != float("Inf") else -1
