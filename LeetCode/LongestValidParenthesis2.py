class Solution(object):
    def longestValidParentheses(self, s):
        open_stack, max_len = [], 0
        start_end_map = dict()
        
        for i in range(len(s)):
            if s[i] == '(':
                open_stack.append(i)
            else:
                if len(open_stack) > 0:
                    last = open_stack.pop()
                    start_end_map[last] = i
        
        curr_max_end, cnt = -1, 0
        
        start = 0
        while start < len(s):
            if start in start_end_map:
                end = start_end_map[start]
                if start > curr_max_end:
                    if start == curr_max_end + 1:
                        cnt += end - start + 1
                    else:
                        max_len = max(max_len, cnt)
                        cnt = end - start + 1
                    curr_max_end = end
                
                start = end+1
            else:
                start += 1
                
        max_len = max(max_len, cnt)
        return max_len
