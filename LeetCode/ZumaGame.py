import heapq

class Solution(object):
    def merge(self, enc, u):
        i = 0
        while 0 <= i < len(enc):
            x, y = enc[i]
            if y >= 3:
                u -= y
                if i-1 >= 0 and i+1 < len(enc) and enc[i-1][0] == enc[i+1][0]:
                    enc = enc[:i-1] + [(enc[i-1][0], enc[i-1][1]+enc[i+1][1])] + enc[i+2:]
                    i -= 1
                else:
                    enc = enc[:i] + enc[i+1:]
            else:
                i += 1
        return enc, u
        
    def findMinStep(self, board, hand):
        encoded, cnt = [], 1
        for i in range(1, len(board)):
            if board[i] != board[i-1]:
                encoded.append((board[i-1], cnt))
                cnt = 1
            else:
                cnt += 1
        
        encoded.append((board[-1], cnt))
                
        heap = [(len(board), encoded, 0, hand)]
        min_balls = float("Inf")
        
        while len(heap) > 0:
            u, enc, num_balls, hnd = heapq.heappop(heap)
            # print u, enc, num_balls, hnd
            
            enc, u = self.merge(enc, u)
                    
            if u <= 0:
                min_balls = min(min_balls, num_balls)
            
            for j in range(len(hnd)):
                h = hnd[j]
                
                flag = False
                for i in range(len(enc)):
                    x, y = enc[i]
                        
                    if x == h:
                        flag = True
                        if y == 2:
                            if i-1 >= 0 and i+1 < len(enc) and enc[i-1][0] == enc[i+1][0]:
                                enc1 = enc[:i-1] + [(enc[i-1][0], enc[i-1][1]+enc[i+1][1])] + enc[i+2:]
                            else:
                                enc1 = enc[:i] + enc[i+1:]
                                
                            heapq.heappush(heap, (u-y, enc1, num_balls+1, hnd[:j] + hnd[j+1:]))
                        else:
                            heapq.heappush(heap, (u+1, enc[:i] + [(x, y+1)] + enc[i+1:], num_balls+1, hnd[:j] + hnd[j+1:]))
                
                if flag is False:
                    heapq.heappush(heap, (u+1, enc + [(h, 1)], num_balls+1, hnd[:j] + hnd[j+1:]))
        
        return min_balls if min_balls != float("Inf") else -1
