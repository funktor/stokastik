import heapq

class Solution(object):
    def longestValidParentheses(self, s):
        open_stack, max_len = [], 0
        min_heap = []
        
        for i in range(len(s)):
            if s[i] == '(':
                open_stack.append(i)
            else:
                if len(open_stack) > 0:
                    last = open_stack.pop()
                    heapq.heappush(min_heap, (last, i))
        
        curr_max_end, cnt = -1, 0
        
        while len(min_heap) > 0:
            start, end = heapq.heappop(min_heap)
            if start > curr_max_end:
                if start == curr_max_end + 1:
                    cnt += end - start + 1
                else:
                    max_len = max(max_len, cnt)
                    cnt = end - start + 1
                curr_max_end = end
                
        max_len = max(max_len, cnt)
        return max_len
